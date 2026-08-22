import json
import logging
from pathlib import Path

import joblib
from pydantic import ValidationError

from app.config import settings
from app.schemas.model_schema import RuntimeModelSettings
from app.utils import uploads_path

logger = logging.getLogger(__name__)


def runtime_settings_path(*, create_parent: bool = False) -> Path:
    return uploads_path("config", create=create_parent).joinpath("model_runtime_settings.json")


def load_runtime_settings() -> RuntimeModelSettings:
    path = runtime_settings_path()
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Older versions used raw grayscale differences on a 0-255 scale.
            baseline_threshold = float(payload.get("baseline_threshold", 0))
            if baseline_threshold > 10 or abs(baseline_threshold - 1.45) < 1e-9:
                payload["baseline_threshold"] = settings.baseline_threshold
            return RuntimeModelSettings(**payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            logger.warning("Ignoring invalid runtime model settings at %s: %s", path, exc)
    return RuntimeModelSettings(
        padim_score_threshold=0.5,
        baseline_threshold=settings.baseline_threshold,
    )


def save_runtime_settings(payload: RuntimeModelSettings) -> RuntimeModelSettings:
    path = runtime_settings_path(create_parent=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return payload


def load_classifier_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        bundle = joblib.load(path)
    except Exception:
        return {}
    return bundle.get("metrics", {}) if isinstance(bundle, dict) else {}


def read_json_artifact(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def artifact_size_mb(*paths: Path | None) -> float:
    size = sum(path.stat().st_size for path in paths if path is not None and path.exists())
    return round(size / (1024 * 1024), 2)


def load_active_classifier_evidence(spec, status: dict, category_metadata: dict) -> tuple[dict, str]:
    """Read metrics from the classifier that the runtime will actually load."""
    engine = status.get("classifier_engine")
    if engine == "fine_tuned_resnet18_onnx" and spec.cnn_classifier_path is not None:
        sidecar = read_json_artifact(spec.cnn_classifier_path.with_suffix(".json"))
        metrics = sidecar.get("metrics") or category_metadata.get("defect_classifier") or {}
        return metrics, "Active ONNX classifier evaluation"
    if engine == "sklearn_feature_classifier" and spec.classifier_path.exists():
        try:
            bundle = joblib.load(spec.classifier_path)
        except Exception:
            return {}, "Classifier artifact could not be read"
        metrics = bundle.get("metrics", {}) if isinstance(bundle, dict) else {}
        return metrics, "Active classifier artifact"
    if engine == "portable_forest":
        return category_metadata.get("defect_classifier") or {}, "Portable classifier evaluation metadata"
    return {}, "No active subtype classifier"


def release_status(binary_f1: float | None, subtype_f1: float | None) -> str:
    if binary_f1 is None or subtype_f1 is None:
        return "Unverified"
    if binary_f1 >= 0.90 and subtype_f1 >= 0.85:
        return "Production"
    if binary_f1 >= 0.90:
        return "Manual review"
    if binary_f1 >= 0.80 and subtype_f1 >= 0.70:
        return "Needs tuning"
    return "Experimental"


def release_reason(binary_f1: float | None, subtype_f1: float | None) -> str:
    status = release_status(binary_f1, subtype_f1)
    if status == "Production":
        return "Binary detection and subtype classification meet the release targets."
    if status == "Manual review":
        return "Binary detection meets target; subtype predictions require operator review."
    if status == "Needs tuning":
        return "Usable for assisted inspection, but one or both model stages remain below target."
    if status == "Experimental":
        return "Model evidence is below the assisted-inspection release threshold."
    return "Complete evaluation metrics are not available for this category."


def build_model_metrics_payload() -> dict:
    from ml.inference import load_model_metadata
    from ml.model_registry import category_model_spec, category_model_statuses

    bottle_spec = category_model_spec("bottle")
    metadata = load_model_metadata(str(bottle_spec.metadata_path))
    classifier_metrics = metadata.get("defect_classifier") or load_classifier_metrics(
        bottle_spec.classifier_path
    )
    classifier_report = classifier_metrics.get("classification_report", {})
    labels = classifier_metrics.get("labels") or metadata.get("defect_classifier", {}).get("labels", [])
    confusion_matrix = classifier_metrics.get("confusion_matrix", [])
    runtime_settings = load_runtime_settings()
    class_rows = {label: row for label, row in classifier_report.items() if isinstance(row, dict) and "f1-score" in row}
    weakest_class = None
    if class_rows:
        weakest_label, weakest_row = min(class_rows.items(), key=lambda item: item[1].get("f1-score", 0))
        weakest_class = {
            "label": weakest_label,
            "precision": weakest_row.get("precision"),
            "recall": weakest_row.get("recall"),
            "f1_score": weakest_row.get("f1-score"),
            "support": weakest_row.get("support"),
        }
    threshold_calibration = {
        "source": "Saved classifier evaluation metrics and active runtime thresholds",
        "eval_size": classifier_metrics.get("eval_size") or metadata.get("defect_classifier", {}).get("eval_size"),
        "accuracy": classifier_metrics.get("accuracy") or metadata.get("defect_classifier", {}).get("accuracy"),
        "macro_f1": classifier_report.get("macro avg", {}).get("f1-score"),
        "weighted_f1": classifier_report.get("weighted avg", {}).get("f1-score"),
        "weakest_class": weakest_class,
        "active_thresholds": {
            "baseline_threshold": runtime_settings.baseline_threshold,
            "review_severity_threshold": runtime_settings.review_severity_threshold,
            "fail_severity_threshold": runtime_settings.fail_severity_threshold,
            "padim_score_threshold": runtime_settings.padim_score_threshold,
        },
        "guidance": [
            "Increase fail severity threshold if too many products are rejected.",
            "Decrease review severity threshold if manual review should catch more borderline defects.",
            "Re-run k-fold validation after changing model features or defect classes.",
        ],
    }

    category_models = []
    baseline_metrics = []
    for status in category_model_statuses(
        settings.use_padim_inference,
        settings.use_openvino_inference,
    ):
        spec = category_model_spec(status["category"])
        category_metadata = read_json_artifact(spec.metadata_path)
        category_metrics = category_metadata.get("metrics") or {}
        if isinstance(category_metrics, list):
            category_metrics = category_metrics[0] if category_metrics else {}
        portable_metrics = category_metadata.get("baseline_threshold_calibration") or {}
        openvino_metrics = category_metadata.get("openvino_spatial_calibration") or {}
        threshold_metrics = category_metadata.get("threshold_calibration") or {}
        recorded_classifier_metrics = category_metadata.get("defect_classifier") or {}
        active_classifier_metrics, subtype_metric_source = load_active_classifier_evidence(
            spec,
            status,
            category_metadata,
        )
        deployed_subtype_metrics = category_metadata.get("deployed_subtype_validation") or {}
        candidate_classifier_metrics = category_metadata.get("defect_classifier_revalidation") or {}
        advanced_active = status["active_engine"].endswith("_openvino") or status["active_engine"] in {
            "padim",
            "patchcore",
        }
        active_detector_metrics = openvino_metrics if status["active_engine"].endswith("_openvino") else portable_metrics
        binary_accuracy = active_detector_metrics.get("accuracy") or active_detector_metrics.get("cv_accuracy")
        binary_precision = active_detector_metrics.get("precision") or active_detector_metrics.get("cv_precision")
        binary_recall = active_detector_metrics.get("recall") or active_detector_metrics.get("cv_recall")
        binary_specificity = active_detector_metrics.get("specificity") or active_detector_metrics.get("cv_specificity")
        binary_balanced_accuracy = active_detector_metrics.get("balanced_accuracy") or active_detector_metrics.get(
            "cv_balanced_accuracy"
        )
        binary_f1 = active_detector_metrics.get("f1") or active_detector_metrics.get("cv_f1")
        binary_auroc = active_detector_metrics.get("auroc")
        if deployed_subtype_metrics:
            subtype_accuracy = deployed_subtype_metrics.get("accuracy")
            subtype_f1 = deployed_subtype_metrics.get("macro_f1")
            subtype_metric_source = "Render-equivalent OpenVINO-mask validation"
        else:
            subtype_accuracy = active_classifier_metrics.get("accuracy")
            subtype_f1 = active_classifier_metrics.get("macro_f1")
        subtype_calibration = deployed_subtype_metrics.get("confidence_calibration") or {}
        subtype_review_threshold = subtype_calibration.get("review_threshold") or status[
            "subtype_confidence_threshold"
        ]
        category_release_status = release_status(binary_f1, subtype_f1)
        if status["active_engine"].endswith("_openvino"):
            binary_metric_source = "Nested validation of active OpenVINO artifact"
        elif status["active_engine"] in {"padim", "patchcore"}:
            binary_metric_source = "Saved checkpoint evaluation"
        else:
            binary_metric_source = "Portable detector cross-validation"
        openvino_export = category_metadata.get("openvino_export") or {}
        detector_artifacts = (
            [spec.openvino_path, spec.openvino_path.with_suffix(".bin")]
            if status["active_engine"].endswith("_openvino") and spec.openvino_path is not None
            else [spec.checkpoint_path]
            if advanced_active
            else [spec.baseline_profile_path, spec.portable_detector_calibrator_path]
        )
        active_classifier_artifact = status.get("artifacts", {}).get("active_classifier") or {}
        active_classifier_size = float(active_classifier_artifact.get("size_bytes", 0)) / (1024 * 1024)
        model_size = artifact_size_mb(*detector_artifacts) + active_classifier_size
        baseline_metrics.append(
            {
                "category": status["category"],
                "available": status["available"],
                "detector": portable_metrics.get("detector", "portable-baseline"),
                "localization": portable_metrics.get("localization", "opencv_normalized_residual"),
                "protocol": portable_metrics.get("protocol"),
                "folds": portable_metrics.get("folds"),
                "samples": portable_metrics.get("samples"),
                "threshold": portable_metrics.get("decision_threshold") or portable_metrics.get("threshold"),
                "residual_threshold": portable_metrics.get("residual_threshold"),
                "accuracy": portable_metrics.get("cv_accuracy"),
                "precision": portable_metrics.get("cv_precision"),
                "recall": portable_metrics.get("cv_recall"),
                "specificity": portable_metrics.get("cv_specificity"),
                "balanced_accuracy": portable_metrics.get("cv_balanced_accuracy"),
                "f1": portable_metrics.get("cv_f1"),
                "auroc": portable_metrics.get("auroc"),
            }
        )
        category_models.append(
            {
                "category": status["category"],
                "available": status["available"],
                "trained": status["trained"],
                "active_engine": status["active_engine"],
                "deployment_tier": status["deployment_tier"],
                "classifier_engine": status["classifier_engine"],
                "subtype_labels": status["subtype_labels"],
                "subtype_count": status["subtype_count"],
                "release_status": category_release_status,
                "release_reason": release_reason(binary_f1, subtype_f1),
                "binary_accuracy": binary_accuracy,
                "binary_precision": binary_precision,
                "binary_recall": binary_recall,
                "binary_specificity": binary_specificity,
                "binary_balanced_accuracy": binary_balanced_accuracy,
                "binary_f1": binary_f1,
                "binary_auroc": binary_auroc,
                "binary_metric_source": binary_metric_source,
                "subtype_accuracy": subtype_accuracy,
                "subtype_macro_f1": subtype_f1,
                "subtype_metric_source": subtype_metric_source,
                "subtype_artifact_accuracy": active_classifier_metrics.get("accuracy"),
                "subtype_artifact_macro_f1": active_classifier_metrics.get("macro_f1"),
                "subtype_deployed_accuracy": deployed_subtype_metrics.get("accuracy"),
                "subtype_deployed_macro_f1": deployed_subtype_metrics.get("macro_f1"),
                "subtype_validation_samples": deployed_subtype_metrics.get("samples"),
                "subtype_validation_protocol": deployed_subtype_metrics.get("protocol")
                or active_classifier_metrics.get("evaluation", {}).get("protocol"),
                "subtype_confidence_calibration": subtype_calibration,
                "subtype_metadata_consistent": (
                    active_classifier_metrics.get("macro_f1") == recorded_classifier_metrics.get("macro_f1")
                ),
                "subtype_confidence_threshold": subtype_review_threshold,
                "input_size": status["input_size"],
                "model_size_mb": round(model_size, 2),
                "model_version": category_metadata.get("model_version", "v1"),
                "deployment_precision": openvino_export.get("precision")
                if status["active_engine"].endswith("_openvino")
                else "FP32",
                "fallback_available": bool(status.get("artifacts", {}).get("profile", {}).get("available")),
                "classification_trained": status["classification_trained"],
                "model_kind": status["model_kind"],
                "decision_threshold": status["decision_threshold"],
                "advanced_decision_threshold": status["decision_threshold"],
                "portable_decision_threshold": portable_metrics.get("decision_threshold")
                or portable_metrics.get("threshold"),
                "image_auroc": category_metrics.get("image_AUROC"),
                "image_f1": category_metrics.get("image_F1Score") or threshold_metrics.get("holdout_f1"),
                "pixel_auroc": category_metrics.get("pixel_AUROC"),
                "calibration_holdout_f1": threshold_metrics.get("holdout_f1"),
                "portable_cv_f1": portable_metrics.get("cv_f1"),
                "portable_cv_balanced_accuracy": portable_metrics.get("cv_balanced_accuracy"),
                "classifier_macro_f1": active_classifier_metrics.get("macro_f1"),
                "classifier_accuracy": active_classifier_metrics.get("accuracy"),
                "classifier_evaluation_protocol": active_classifier_metrics.get("evaluation", {}).get("protocol"),
                "candidate_classifier_macro_f1": candidate_classifier_metrics.get("macro_f1"),
                "candidate_classifier_accuracy": candidate_classifier_metrics.get("accuracy"),
                "active_classifier_macro_f1": active_classifier_metrics.get("macro_f1"),
                "active_classifier_accuracy": active_classifier_metrics.get("accuracy"),
                "openvino_accuracy": openvino_metrics.get("accuracy"),
                "openvino_balanced_accuracy": openvino_metrics.get("balanced_accuracy"),
                "openvino_f1": openvino_metrics.get("f1"),
                "openvino_auroc": openvino_metrics.get("auroc"),
                "openvino_recall": openvino_metrics.get("recall"),
                "openvino_specificity": openvino_metrics.get("specificity"),
                "trained_at": category_metadata.get("trained_at"),
            }
        )

    measured_binary_f1 = [row["binary_f1"] for row in category_models if row["binary_f1"] is not None]
    measured_subtype_f1 = [row["subtype_macro_f1"] for row in category_models if row["subtype_macro_f1"] is not None]
    model_comparison = [
        {
            "name": "Active Good / Defective detectors",
            "task": "Category-specific anomaly detection and localization",
            "framework": "OpenVINO or portable ResNet18 normal memory",
            "primary_metric": "Mean active F1",
            "score": sum(measured_binary_f1) / len(measured_binary_f1) if measured_binary_f1 else None,
            "secondary_metric": "Production categories",
            "secondary_score": sum(row["release_status"] == "Production" for row in category_models),
            "status": "deployed-runtime",
        },
        {
            "name": "Active defect subtype classifiers",
            "task": "Category-specific defect type classification",
            "framework": "ONNX CNN or scikit-learn feature classifier",
            "primary_metric": "Mean macro F1",
            "score": sum(measured_subtype_f1) / len(measured_subtype_f1) if measured_subtype_f1 else None,
            "secondary_metric": "Verified categories",
            "secondary_score": len(measured_subtype_f1),
            "status": "deployed-runtime",
        },
        {
            "name": "Portable fallback coverage",
            "task": "Render-compatible inspection when advanced artifacts are unavailable",
            "framework": "OpenCV DNN / NumPy / scikit-learn",
            "primary_metric": "Categories ready",
            "score": sum(row["available"] for row in category_models),
            "secondary_metric": "Advanced categories",
            "secondary_score": sum(row["deployment_tier"] == "advanced" for row in category_models),
            "status": "fallback-runtime",
        },
    ]

    return {
        "metadata": metadata,
        "runtime_settings": load_runtime_settings().model_dump(),
        "model_comparison": model_comparison,
        "classifier_report": classifier_report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix,
            "description": "Rows are actual labels; columns are predicted labels.",
        },
        "threshold_calibration": threshold_calibration,
        "baseline_metrics": baseline_metrics,
        "category_models": category_models,
        "release_summary": {
            "categories": len(category_models),
            "production": sum(row["release_status"] == "Production" for row in category_models),
            "manual_review": sum(row["release_status"] == "Manual review" for row in category_models),
            "needs_tuning": sum(row["release_status"] == "Needs tuning" for row in category_models),
            "experimental": sum(row["release_status"] == "Experimental" for row in category_models),
            "unverified": sum(row["release_status"] == "Unverified" for row in category_models),
            "binary_target_met": sum(
                row["binary_f1"] is not None and row["binary_f1"] >= 0.90 for row in category_models
            ),
            "subtype_target_met": sum(
                row["subtype_macro_f1"] is not None and row["subtype_macro_f1"] >= 0.85
                for row in category_models
            ),
            "openvino": sum(row["active_engine"].endswith("_openvino") for row in category_models),
            "advanced": sum(row["deployment_tier"] == "advanced" for row in category_models),
            "portable": sum(row["deployment_tier"] == "portable" for row in category_models),
        },
    }
