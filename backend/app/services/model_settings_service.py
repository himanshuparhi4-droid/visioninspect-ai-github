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


def uploads_path(*parts: str) -> Path:
    path = resolve_backend_path(settings.upload_dir).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_settings_path() -> Path:
    return uploads_path("config").joinpath("model_runtime_settings.json")


def load_runtime_settings() -> RuntimeModelSettings:
    path = runtime_settings_path()
    if path.exists():
        try:
            return RuntimeModelSettings(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return RuntimeModelSettings(
        padim_score_threshold=0.5,
        baseline_threshold=settings.baseline_threshold,
    )


def save_runtime_settings(payload: RuntimeModelSettings) -> RuntimeModelSettings:
    path = runtime_settings_path()
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
            "score": load_runtime_settings().baseline_threshold,
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
        },
    }
