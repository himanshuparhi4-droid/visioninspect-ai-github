from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.config import settings
from app.services.cloudinary_service import upload_image_or_local_url

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PredictionError(RuntimeError):
    pass


def resolve_backend_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (BACKEND_DIR / path).resolve()


def uploads_path(*parts: str) -> Path:
    path = resolve_backend_path(settings.upload_dir).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model_metadata() -> dict:
    from ml.inference import load_model_metadata as load_ml_model_metadata

    return load_ml_model_metadata(str(resolve_backend_path(settings.model_metadata_path)))


def build_inference_config():
    from app.services.model_settings_service import load_runtime_settings
    from ml.inference import InferenceConfig

    runtime_settings = load_runtime_settings()
    return InferenceConfig(
        use_padim_inference=settings.use_padim_inference,
        padim_inference_accelerator=settings.padim_inference_accelerator,
        model_checkpoint_path=resolve_backend_path(settings.model_checkpoint_path),
        classifier_model_path=resolve_backend_path(settings.classifier_model_path),
        model_metadata_path=resolve_backend_path(settings.model_metadata_path),
        baseline_reference_path=resolve_backend_path(settings.baseline_reference_path),
        baseline_profile_path=resolve_backend_path(settings.baseline_profile_path),
        baseline_threshold=runtime_settings.baseline_threshold,
        padim_score_threshold=runtime_settings.padim_score_threshold,
        review_severity_threshold=runtime_settings.review_severity_threshold,
        fail_severity_threshold=runtime_settings.fail_severity_threshold,
    )


def save_visual_outputs(processed_image: np.ndarray, heatmap_image: np.ndarray) -> dict:
    output_stem = uuid4().hex

    processed_path = uploads_path("processed").joinpath(f"{output_stem}_processed.png")
    cv2.imwrite(str(processed_path), np.clip(processed_image, 0, 255).astype(np.uint8))

    heatmap_path = uploads_path("heatmaps").joinpath(f"{output_stem}_heatmap.png")
    cv2.imwrite(str(heatmap_path), np.clip(heatmap_image, 0, 255).astype(np.uint8))

    return {
        "processed_image_path": str(processed_path),
        "processed_image_url": upload_image_or_local_url(processed_path, "processed"),
        "heatmap_path": str(heatmap_path),
        "heatmap_url": upload_image_or_local_url(heatmap_path, "heatmaps"),
    }


def inspect_image_file(image_path: str | Path) -> dict:
    from ml.inference import InferenceError, inspect_image

    try:
        result = inspect_image(image_path, build_inference_config())
    except InferenceError as exc:
        raise PredictionError(str(exc)) from exc

    outputs = save_visual_outputs(result.pop("processed_image"), result.pop("heatmap_image"))
    result.pop("anomaly_map", None)
    result.pop("pred_mask", None)
    return {**result, **outputs}


__all__ = [
    "PredictionError",
    "inspect_image_file",
    "load_model_metadata",
    "resolve_backend_path",
    "uploads_path",
]
