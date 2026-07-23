import json
from pathlib import Path

import joblib

from app.config import settings
from app.schemas.model_schema import RuntimeModelSettings

BACKEND_DIR = Path(__file__).resolve().parents[2]


def resolve_backend_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (BACKEND_DIR / path).resolve()


def uploads_path(*parts: str, create: bool = True) -> Path:
    path = resolve_backend_path(settings.upload_dir).joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_settings_path(*, create_parent: bool = False) -> Path:
    return uploads_path("config", create=create_parent).joinpath("model_runtime_settings.json")


def load_runtime_settings() -> RuntimeModelSettings:
    path = runtime_settings_path()
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Older versions used raw grayscale differences on a 0-255 scale.
            if float(payload.get("baseline_threshold", 0)) > 10:
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
    from app.services.prediction_service import load_model_metadata

    metadata = load_model_metadata()
    classifier_metrics = load_classifier_metrics()
    padim_metrics = metadata.get("metrics", {})
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

    model_comparison = [
        {
            "name": "PaDiM anomaly detector",
            "task": "Good vs defective anomaly detection and heatmap localization",
            "framework": "PyTorch / Anomalib",
            "primary_metric": "Image AUROC",
            "score": padim_metrics.get("image_AUROC"),
            "secondary_metric": "Pixel AUROC",
            "secondary_score": padim_metrics.get("pixel_AUROC"),
            "status": "production-serving",
        },
        {
            "name": "ResNet18 feature classifier",
            "task": "Defect type classification",
            "framework": "PyTorch features / scikit-learn",
            "primary_metric": "Accuracy",
            "score": classifier_metrics.get("accuracy") or metadata.get("defect_classifier", {}).get("accuracy"),
            "secondary_metric": "Macro F1",
            "secondary_score": classifier_report.get("macro avg", {}).get("f1-score"),
            "status": "type-classifier",
        },
        {
            "name": "OpenCV baseline",
            "task": "Fallback anomaly scoring and visual heatmap",
            "framework": "OpenCV / NumPy",
            "primary_metric": "Threshold",
            "score": runtime_settings.baseline_threshold,
            "secondary_metric": "Purpose",
            "secondary_score": "fallback",
            "status": "backup",
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
    }
