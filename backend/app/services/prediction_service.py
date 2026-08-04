from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.config import settings
from app.services.cloudinary_service import cleanup_stored_image, upload_image_or_local_url
from app.utils import resolve_backend_path, uploads_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PredictionError(RuntimeError):
    pass


def load_model_metadata() -> dict:
    from ml.inference import load_model_metadata as load_ml_model_metadata

    return load_ml_model_metadata(str(resolve_backend_path(settings.model_metadata_path)))


def build_inference_config(category: str, critical_zones: tuple[str, ...] = ()):
    from app.services.model_settings_service import load_runtime_settings
    from ml.inference import InferenceConfig
    from ml.model_registry import CategoryModelError, category_model_spec, is_valid_checkpoint

    runtime_settings = load_runtime_settings()
    try:
        spec = category_model_spec(category)
    except CategoryModelError as exc:
        raise PredictionError(str(exc)) from exc
    if not spec.is_runnable:
        raise PredictionError(
            f"The {spec.category} runtime artifacts are unavailable. "
            "Restore its normal profile and model metadata before inspection."
        )
    checkpoint_path = (
        resolve_backend_path(settings.model_checkpoint_path) if spec.category == "bottle" else spec.checkpoint_path
    )
    advanced_model_available = is_valid_checkpoint(checkpoint_path) or (
        spec.openvino_path is not None
        and spec.openvino_path.exists()
        and spec.openvino_path.with_suffix(".bin").exists()
    )
    use_advanced_model = settings.use_padim_inference and advanced_model_available
    use_openvino = (
        use_advanced_model
        and settings.use_openvino_inference
        and spec.openvino_path is not None
        and spec.openvino_path.exists()
    )
    return InferenceConfig(
        category=spec.category,
        anomaly_model_kind=spec.model_kind,
        use_padim_inference=use_advanced_model,
        use_openvino_inference=use_openvino,
        openvino_inference_device=settings.openvino_inference_device,
        padim_inference_accelerator=settings.padim_inference_accelerator,
        model_checkpoint_path=checkpoint_path,
        classifier_model_path=spec.classifier_path,
        cnn_classifier_model_path=spec.cnn_classifier_path,
        model_metadata_path=spec.metadata_path,
        baseline_profile_path=spec.baseline_profile_path,
        # The dashboard value remains a global sensitivity control while each
        # product category keeps its own calibrated portable-model threshold.
        baseline_threshold=max(
            0.0,
            spec.baseline_score_threshold * runtime_settings.baseline_threshold / 1.34,
        ),
        baseline_residual_threshold=spec.baseline_residual_threshold,
        # 0.50 is the neutral dashboard setting. Moving it shifts every
        # calibrated category threshold by the same small operational offset.
        padim_score_threshold=max(
            0.0, min(1.0, spec.padim_score_threshold + runtime_settings.padim_score_threshold - 0.5)
        ),
        review_severity_threshold=runtime_settings.review_severity_threshold,
        fail_severity_threshold=runtime_settings.fail_severity_threshold,
        critical_zones=critical_zones,
        openvino_path=spec.openvino_path if use_openvino else None,
    )


def save_visual_outputs(processed_image: np.ndarray, heatmap_image: np.ndarray) -> dict:
    output_stem = uuid4().hex
    processed_path = uploads_path("processed").joinpath(f"{output_stem}_processed.png")
    heatmap_path = uploads_path("heatmaps").joinpath(f"{output_stem}_heatmap.png")
    processed_url = None
    heatmap_url = None
    try:
        if not cv2.imwrite(str(processed_path), np.clip(processed_image, 0, 255).astype(np.uint8)):
            raise PredictionError("Could not save the processed inspection image")
        processed_url = upload_image_or_local_url(processed_path, "processed")
        if not cv2.imwrite(str(heatmap_path), np.clip(heatmap_image, 0, 255).astype(np.uint8)):
            raise PredictionError("Could not save the defect heatmap")
        heatmap_url = upload_image_or_local_url(heatmap_path, "heatmaps")
        return {
            "processed_image_path": str(processed_path),
            "processed_image_url": processed_url,
            "heatmap_path": str(heatmap_path),
            "heatmap_url": heatmap_url,
        }
    except Exception:
        cleanup_stored_image(processed_path, processed_url)
        cleanup_stored_image(heatmap_path, heatmap_url)
        raise


def inspect_image_file(
    image_path: str | Path,
    category: str,
    critical_zones: tuple[str, ...] = (),
) -> dict:
    from ml.inference import InferenceError, inspect_image

    try:
        result = inspect_image(image_path, build_inference_config(category, critical_zones))
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
]
