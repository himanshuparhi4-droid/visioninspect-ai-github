from __future__ import annotations

import gc
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np

from app.config import resource_constrained_runtime, settings
from app.services.cloudinary_service import cleanup_stored_image, upload_image_or_local_url
from app.utils import uploads_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)
storage_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inspection-storage")
inference_lock = Lock()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PredictionError(RuntimeError):
    pass


def build_inference_config(category: str, critical_zones: tuple[str, ...] = ()):
    from app.services.model_settings_service import (
        load_active_classifier_evidence,
        load_runtime_settings,
        read_json_artifact,
    )
    from ml.inference import InferenceConfig
    from ml.model_registry import (
        CategoryModelError,
        category_model_spec,
        classifier_runtime_status,
        is_valid_checkpoint,
        openvino_runtime_is_memory_safe,
    )

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
    checkpoint_path = spec.checkpoint_path
    advanced_model_available = is_valid_checkpoint(checkpoint_path) or (
        spec.openvino_path is not None
        and spec.openvino_path.exists()
        and spec.openvino_path.with_suffix(".bin").exists()
    )
    constrained_runtime = resource_constrained_runtime()
    classifier_status = classifier_runtime_status(spec)
    use_advanced_model = (
        settings.use_padim_inference
        and advanced_model_available
        and not constrained_runtime
    )
    use_openvino = (
        settings.use_openvino_inference
        and spec.openvino_path is not None
        and spec.openvino_path.exists()
        and spec.openvino_path.with_suffix(".bin").exists()
        and (
            not constrained_runtime
            or openvino_runtime_is_memory_safe(spec, classifier_status["engine"])
        )
    )
    category_metadata = read_json_artifact(spec.metadata_path)
    subtype_metrics, _ = load_active_classifier_evidence(spec, classifier_status, category_metadata)
    deployed_subtype_validation = category_metadata.get("deployed_subtype_validation") or {}
    confidence_calibration = deployed_subtype_validation.get("confidence_calibration") or None
    calibrated_review_threshold = (
        confidence_calibration.get("review_threshold") if confidence_calibration else None
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
        subtype_confidence_threshold=float(calibrated_review_threshold or spec.subtype_confidence_threshold),
        portable_threshold_scale=max(0.25, min(2.0, runtime_settings.baseline_threshold / 1.34)),
        critical_zones=critical_zones,
        openvino_path=spec.openvino_path if use_openvino else None,
        openvino_calibrator_path=spec.openvino_calibrator_path if use_openvino else None,
        portable_detector_calibrator_path=spec.portable_detector_calibrator_path,
        compact_classifier_path=spec.compact_classifier_path,
        input_size=spec.input_size,
        subtype_model_macro_f1=(
            deployed_subtype_validation.get("macro_f1")
            if deployed_subtype_validation
            else subtype_metrics.get("macro_f1")
        ),
        subtype_confidence_calibration=confidence_calibration,
        release_detector_before_classification=constrained_runtime,
    )


def release_inference_runtimes() -> None:
    """Return native model memory to the constrained deployment between requests."""
    from ml.defect_classifier import release_classifier_runtimes
    from ml.inference import load_model_metadata, load_normal_profile
    from ml.padim_detector import release_anomaly_runtimes

    release_anomaly_runtimes()
    release_classifier_runtimes()
    load_model_metadata.cache_clear()
    load_normal_profile.cache_clear()
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        except (AttributeError, OSError):
            logger.debug("Native heap trimming is unavailable", exc_info=True)


def process_rss_mb() -> float | None:
    """Read current Linux resident memory without adding a monitoring dependency."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        return None


def save_visual_outputs(processed_image: np.ndarray, heatmap_image: np.ndarray) -> dict:
    output_stem = uuid4().hex
    processed_path = uploads_path("processed").joinpath(f"{output_stem}_processed.png")
    heatmap_path = uploads_path("heatmaps").joinpath(f"{output_stem}_heatmap.png")
    processed_url = None
    heatmap_url = None
    try:
        if not cv2.imwrite(str(processed_path), np.clip(processed_image, 0, 255).astype(np.uint8)):
            raise PredictionError("Could not save the processed inspection image")
        if not cv2.imwrite(str(heatmap_path), np.clip(heatmap_image, 0, 255).astype(np.uint8)):
            raise PredictionError("Could not save the defect heatmap")
        futures = {
            "processed": storage_executor.submit(upload_image_or_local_url, processed_path, "processed"),
            "heatmap": storage_executor.submit(upload_image_or_local_url, heatmap_path, "heatmaps"),
        }
        errors = []
        for name, future in futures.items():
            try:
                if name == "processed":
                    processed_url = future.result()
                else:
                    heatmap_url = future.result()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
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

    started_at = perf_counter()
    constrained_runtime = resource_constrained_runtime()
    lock_context = inference_lock if constrained_runtime else nullcontext()
    with lock_context:
        try:
            result = inspect_image(image_path, build_inference_config(category, critical_zones))
        except InferenceError as exc:
            raise PredictionError(str(exc)) from exc
        finally:
            if constrained_runtime:
                release_inference_runtimes()

    inference_ms = (perf_counter() - started_at) * 1000
    storage_started_at = perf_counter()
    outputs = save_visual_outputs(result.pop("processed_image"), result.pop("heatmap_image"))
    storage_ms = (perf_counter() - storage_started_at) * 1000
    total_ms = (perf_counter() - started_at) * 1000
    rss_mb = process_rss_mb()
    result.setdefault("explainability", {})["runtime_ms"] = {
        "inference": round(inference_ms, 1),
        "visual_storage": round(storage_ms, 1),
        "total": round(total_ms, 1),
    }
    logger.info(
        "inspection_timing category=%s inference_ms=%.1f visual_storage_ms=%.1f total_ms=%.1f rss_mb=%s",
        category,
        inference_ms,
        storage_ms,
        total_ms,
        round(rss_mb, 1) if rss_mb is not None else "unavailable",
    )
    result.pop("anomaly_map", None)
    result.pop("pred_mask", None)
    return {**result, **outputs}


def warm_category_runtime(category: str) -> dict:
    """Load one category's detector and subtype runtime before inspection."""
    from ml.defect_classifier import (
        load_classifier_runtime,
        load_portable_cnn_runtime,
        shared_feature_runtime,
    )
    from ml.inference import load_normal_profile
    from ml.model_registry import category_model_spec, classifier_runtime_status
    from ml.padim_detector import load_anomaly_runtime, load_openvino_runtime

    started_at = perf_counter()
    config = build_inference_config(category)
    spec = category_model_spec(category)
    warnings: list[str] = []

    if config.use_openvino_inference and config.openvino_path is not None:
        load_openvino_runtime(str(config.openvino_path), config.openvino_inference_device)
        detector_engine = f"{config.anomaly_model_kind}_openvino"
    elif config.use_padim_inference:
        load_anomaly_runtime(
            str(config.model_checkpoint_path),
            config.anomaly_model_kind,
            config.padim_inference_accelerator,
        )
        detector_engine = config.anomaly_model_kind
    else:
        load_normal_profile(str(config.baseline_profile_path))
        if config.portable_detector_calibrator_path and config.portable_detector_calibrator_path.exists():
            from ml.classifier import load_portable_forest

            calibrator_path = config.portable_detector_calibrator_path
            load_portable_forest(str(calibrator_path), calibrator_path.stat().st_mtime_ns)
        detector_engine = "portable_baseline"

    classifier = classifier_runtime_status(spec)
    classifier_engine = str(classifier["engine"])
    try:
        if classifier_engine == "fine_tuned_resnet18_onnx" and spec.cnn_classifier_path is not None:
            load_portable_cnn_runtime(
                str(spec.cnn_classifier_path),
                str(spec.cnn_classifier_path.with_suffix(".json")),
            )
        elif classifier_engine == "portable_forest" and spec.compact_classifier_path is not None:
            from ml.classifier import load_portable_forest

            compact_path = spec.compact_classifier_path
            load_portable_forest(str(compact_path), compact_path.stat().st_mtime_ns)
        elif classifier_engine == "sklearn_feature_classifier":
            load_classifier_runtime(str(spec.classifier_path))
            shared_feature_runtime()
    except Exception as exc:
        warnings.append(str(exc))
        classifier_engine = "fallback_on_first_inference"

    return {
        "category": spec.category,
        "ready": not warnings,
        "detector_engine": detector_engine,
        "classifier_engine": classifier_engine,
        "warnings": warnings,
        "warmup_ms": round((perf_counter() - started_at) * 1000, 1),
    }


__all__ = [
    "PredictionError",
    "inspect_image_file",
    "warm_category_runtime",
]
