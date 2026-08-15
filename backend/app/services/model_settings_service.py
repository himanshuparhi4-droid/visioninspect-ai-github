import json
from pathlib import Path

import joblib

from app.config import settings
from app.schemas.model_schema import RuntimeModelSettings
from app.utils import resolve_backend_path, uploads_path


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
        except Exception:
            pass
    return RuntimeModelSettings(
        padim_score_threshold=0.5,
        baseline_threshold=settings.baseline_threshold,
    )


def save_runtime_settings(payload: RuntimeModelSettings) -> RuntimeModelSettings:
    path = runtime_settings_path(create_parent=True)
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return payload


def load_classifier_metrics() -> dict:
    path = resolve_backend_path(settings.classifier_model_path)
    if not path.exists():
        return {}
    try:
        bundle = joblib.load(path)
    except Exception:
        return {}
    return bundle.get("metrics", {}) if isinstance(bundle, dict) else {}


def build_model_metrics_payload() -> dict:
    from ml.inference import load_model_metadata
    from ml.model_registry import category_model_spec, category_model_statuses

    metadata = load_model_metadata(str(resolve_backend_path(settings.model_metadata_path)))
    classifier_metrics = load_classifier_metrics()
    padim_metrics = metadata.get("metrics", {})
    if isinstance(padim_metrics, list):
        padim_metrics = padim_metrics[0] if padim_metrics else {}
    classifier_report = classifier_metrics.get("classification_report", {})
    baseline_calibration = metadata.get("baseline_threshold_calibration", {})
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

    model_comparison = [
        {
            "name": "Advanced category anomaly detector",
            "task": "Good vs defective anomaly detection and heatmap localization",
            "framework": "PyTorch / Anomalib (PaDiM or PatchCore)",
            "primary_metric": "Image AUROC",
            "score": padim_metrics.get("image_AUROC"),
            "secondary_metric": "Pixel AUROC",
            "secondary_score": padim_metrics.get("pixel_AUROC"),
            "status": "optional-advanced-runtime",
        },
        {
            "name": "ResNet18 + texture classifier",
            "task": "Defect type classification",
            "framework": "PyTorch features / OpenCV descriptors / scikit-learn",
            "primary_metric": "Accuracy",
            "score": classifier_metrics.get("accuracy") or metadata.get("defect_classifier", {}).get("accuracy"),
            "secondary_metric": "Macro F1",
            "secondary_score": classifier_report.get("macro avg", {}).get("f1-score"),
            "status": "type-classifier",
        },
        {
            "name": "Portable normal-memory baseline",
            "task": "Good/defective screening and residual heatmap",
            "framework": "ResNet18 normal memory / OpenCV localization",
            "primary_metric": "CV balanced accuracy",
            "score": baseline_calibration.get("cv_balanced_accuracy"),
            "secondary_metric": "CV F1",
            "secondary_score": baseline_calibration.get("cv_f1"),
            "status": "default-runtime",
        },
    ]
    category_models = []
    baseline_metrics = []
    for status in category_model_statuses(
        settings.use_padim_inference,
        settings.use_openvino_inference,
    ):
        spec = category_model_spec(status["category"])
        category_metadata = {}
        if spec.metadata_path.exists():
            try:
                category_metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                category_metadata = {}
        category_metrics = category_metadata.get("metrics") or {}
        if isinstance(category_metrics, list):
            category_metrics = category_metrics[0] if category_metrics else {}
        portable_metrics = category_metadata.get("baseline_threshold_calibration") or {}
        openvino_metrics = category_metadata.get("openvino_spatial_calibration") or {}
        threshold_metrics = category_metadata.get("threshold_calibration") or {}
        active_classifier_metrics = category_metadata.get("defect_classifier") or {}
        classifier_metrics = category_metadata.get("defect_classifier_revalidation") or active_classifier_metrics
        baseline_metrics.append(
            {
                "category": status["category"],
                "available": status["available"],
                "detector": portable_metrics.get("detector", "portable-baseline"),
                "localization": portable_metrics.get("localization", "opencv_normalized_residual"),
                "protocol": portable_metrics.get("protocol"),
                "folds": portable_metrics.get("folds"),
                "samples": portable_metrics.get("samples"),
                "threshold": portable_metrics.get("threshold"),
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
                "classification_trained": status["classification_trained"],
                "model_kind": status["model_kind"],
                "decision_threshold": status["decision_threshold"],
                "advanced_decision_threshold": status["decision_threshold"],
                "portable_decision_threshold": portable_metrics.get("threshold"),
                "image_auroc": category_metrics.get("image_AUROC"),
                "image_f1": category_metrics.get("image_F1Score") or threshold_metrics.get("holdout_f1"),
                "pixel_auroc": category_metrics.get("pixel_AUROC"),
                "calibration_holdout_f1": threshold_metrics.get("holdout_f1"),
                "portable_cv_f1": portable_metrics.get("cv_f1"),
                "portable_cv_balanced_accuracy": portable_metrics.get("cv_balanced_accuracy"),
                "classifier_macro_f1": classifier_metrics.get("macro_f1"),
                "classifier_accuracy": classifier_metrics.get("accuracy"),
                "classifier_evaluation_protocol": classifier_metrics.get("evaluation", {}).get("protocol"),
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
    }
