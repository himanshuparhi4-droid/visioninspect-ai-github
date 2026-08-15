from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np


class PadimInferenceError(RuntimeError):
    pass


ANOMALY_MODEL_CACHE_SIZE = max(1, min(int(os.getenv("ANOMALY_MODEL_CACHE_SIZE", "2")), 4))
OPENVINO_INFERENCE_LOCK = Lock()


def choose_accelerator(configured: str = "auto") -> str:
    configured = configured.lower()
    if configured != "auto":
        return configured

    try:
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=ANOMALY_MODEL_CACHE_SIZE)
def load_anomaly_runtime(checkpoint_path: str, model_kind: str = "padim", accelerator: str = "auto") -> tuple[object, object]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise PadimInferenceError(f"PaDiM checkpoint not found: {path}")

    try:

        def build_runtime():
            import torch
            from anomalib.engine import Engine
            from anomalib.models import Padim, Patchcore

            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

            model_class = {"padim": Padim, "patchcore": Patchcore}.get(model_kind.lower())
            if model_class is None:
                raise PadimInferenceError(f"Unsupported anomaly model '{model_kind}'")
            # These checkpoints are produced locally by our Anomalib training scripts and
            # contain the serialized preprocessing configuration. PyTorch 2.6 defaults to
            # `weights_only=True`, which rejects that trusted configuration at load time.
            return (
                model_class.load_from_checkpoint(path, weights_only=False),
                Engine(
                    accelerator=choose_accelerator(accelerator),
                    devices=1,
                    logger=False,
                    enable_progress_bar=False,
                ),
            )

        model, engine = build_runtime()
    except Exception as exc:
        raise PadimInferenceError(f"Could not load {model_kind} checkpoint") from exc

    return model, engine


@lru_cache(maxsize=4)
def load_openvino_runtime(model_path: str, device: str = "CPU") -> object:
    path = Path(model_path)
    if not path.exists():
        raise PadimInferenceError(f"OpenVINO model not found: {path}")

    try:
        import openvino as ov

        compile_config = {}
        if device.upper() == "CPU":
            cpu_threads = max(1, int(os.getenv("OPENVINO_CPU_THREADS", "1")))
            compile_config = {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
                "INFERENCE_NUM_THREADS": cpu_threads,
                "ENABLE_CPU_PINNING": False,
            }
        return ov.Core().compile_model(str(path), device, compile_config)
    except Exception as exc:
        raise PadimInferenceError(f"Could not load OpenVINO model on {device}") from exc


@lru_cache(maxsize=8)
def load_openvino_calibrator(path_value: str) -> dict:
    path = Path(path_value)
    if not path.exists():
        raise PadimInferenceError(f"OpenVINO calibrator not found: {path}")
    try:
        from ml.classifier import load_portable_forest

        return load_portable_forest(str(path), path.stat().st_mtime_ns)
    except Exception as exc:
        raise PadimInferenceError(f"Could not load OpenVINO calibrator: {path}") from exc


def openvino_spatial_features(score: float, anomaly_map: np.ndarray) -> np.ndarray:
    """Summarize global and local anomaly-map evidence for image calibration."""
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32).squeeze()
    features = [
        float(score),
        float(anomaly_map.mean()),
        float(anomaly_map.std()),
        float(anomaly_map.min()),
        float(anomaly_map.max()),
    ]
    features.extend(np.percentile(anomaly_map, [50, 75, 85, 90, 95, 97, 98, 99, 99.5, 99.9]).tolist())
    features.extend(
        float((anomaly_map > threshold).mean())
        for threshold in (0.40, 0.45, 0.47, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70)
    )
    height, width = anomaly_map.shape
    for row in range(4):
        for column in range(4):
            cell = anomaly_map[
                row * height // 4 : (row + 1) * height // 4,
                column * width // 4 : (column + 1) * width // 4,
            ]
            features.extend(
                [
                    float(cell.mean()),
                    float(np.percentile(cell, 95)),
                    float(cell.max()),
                ]
            )
    return np.asarray(features, dtype=np.float32)


def calibrated_defect_probability(
    calibrator_path: str | Path, score: float, anomaly_map: np.ndarray
) -> tuple[float, float]:
    from ml.classifier import predict_portable_forest

    bundle = load_openvino_calibrator(str(calibrator_path))
    features = openvino_spatial_features(score, anomaly_map).reshape(1, -1)
    prediction = predict_portable_forest(calibrator_path, features)
    probabilities = prediction["probabilities"][0]
    classes = prediction["classes"]
    positive = np.flatnonzero(classes.astype(bool))
    if not len(positive):
        raise PadimInferenceError("OpenVINO calibrator has no defective class")
    return float(probabilities[int(positive[0])]), float(bundle["decision_threshold"])


def predict_with_openvino(
    image_path: Path,
    model_path: str | Path,
    *,
    model_kind: str,
    score_threshold: float,
    device: str,
    calibrator_path: str | Path | None,
) -> dict:
    import cv2

    compiled_model = load_openvino_runtime(str(model_path), device)
    image = cv2.imread(str(image_path))
    if image is None:
        raise PadimInferenceError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
    tensor = image.transpose(2, 0, 1)[None]
    try:
        with OPENVINO_INFERENCE_LOCK:
            outputs = compiled_model([tensor])
        score = float(np.asarray(outputs[compiled_model.output("pred_score")]).reshape(-1)[0])
        anomaly_map = np.asarray(outputs[compiled_model.output("anomaly_map")]).squeeze().astype(np.float32)
        pred_mask = np.asarray(outputs[compiled_model.output("pred_mask")]).squeeze().astype(bool)
    except Exception as exc:
        raise PadimInferenceError("OpenVINO prediction failed") from exc

    defect_probability = None
    calibration_threshold = None
    if calibrator_path and Path(calibrator_path).exists():
        defect_probability, calibration_threshold = calibrated_defect_probability(calibrator_path, score, anomaly_map)
        is_defective = defect_probability >= calibration_threshold
        detection_confidence = max(defect_probability, 1.0 - defect_probability)
        decision_basis = "spatial_calibrator"
    else:
        is_defective = score > score_threshold
        detection_confidence = padim_detection_confidence(score, score_threshold, is_defective)
        decision_basis = "score_threshold"

    if is_defective and not np.any(pred_mask):
        pred_mask = anomaly_map >= np.percentile(anomaly_map, 99)

    return {
        "engine": f"{model_kind}_openvino",
        "anomaly_score": round(score, 6),
        "decision_threshold": score_threshold,
        "decision_basis": decision_basis,
        "calibrated_defect_probability": round(defect_probability, 6) if defect_probability is not None else None,
        "calibration_threshold": round(calibration_threshold, 6) if calibration_threshold is not None else None,
        "is_defective": bool(is_defective),
        "detection_confidence": round(float(detection_confidence), 4),
        "anomaly_map": anomaly_map,
        "pred_mask": pred_mask,
        "fallback_used": False,
        "fallback_reason": None,
    }


def tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def padim_detection_confidence(score: float, threshold: float, is_defective: bool) -> float:
    margin = abs(score - threshold)
    confidence = 0.58 + min(0.40, margin * 2.0)
    if is_defective and score >= 0.80:
        confidence = max(confidence, 0.90)
    return round(float(max(0.55, min(0.99, confidence))), 4)


def predict_with_anomaly_model(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    model_kind: str = "padim",
    score_threshold: float = 0.5,
    accelerator: str = "auto",
    openvino_path: str | Path | None = None,
    openvino_device: str = "CPU",
    openvino_calibrator_path: str | Path | None = None,
) -> dict:
    model_kind = model_kind.lower()
    image_path = Path(image_path)
    if not image_path.exists():
        raise PadimInferenceError(f"Image not found: {image_path}")
    try:
        import cv2

        if cv2.imread(str(image_path)) is None:
            raise PadimInferenceError(f"Could not read image: {image_path}")
    except PadimInferenceError:
        raise
    except Exception as exc:
        raise PadimInferenceError(f"Could not validate image: {image_path}") from exc

    if openvino_path and Path(openvino_path).exists():
        try:
            return predict_with_openvino(
                image_path,
                openvino_path,
                model_kind=model_kind,
                score_threshold=score_threshold,
                device=openvino_device,
                calibrator_path=openvino_calibrator_path,
            )
        except Exception as exc:
            if not (checkpoint_path and Path(checkpoint_path).exists()):
                raise PadimInferenceError("OpenVINO prediction failed") from exc

    model, engine = load_anomaly_runtime(str(checkpoint_path), model_kind, accelerator)

    try:
        predictions = engine.predict(
            model=model,
            data_path=Path(image_path),
            return_predictions=True,
        )
    except Exception as exc:
        raise PadimInferenceError(f"{model_kind} prediction failed") from exc

    if not predictions:
        raise PadimInferenceError(f"{model_kind} returned no predictions")

    prediction = predictions[0]
    score = float(tensor_to_numpy(prediction.pred_score).reshape(-1)[0])
    is_defective = score > score_threshold
    anomaly_map = tensor_to_numpy(prediction.anomaly_map).squeeze().astype(np.float32)

    if hasattr(prediction, "pred_mask") and prediction.pred_mask is not None:
        if isinstance(prediction.pred_mask, np.ndarray):
            pred_mask = prediction.pred_mask.squeeze().astype(bool)
        else:
            pred_mask = tensor_to_numpy(prediction.pred_mask).squeeze().astype(bool)
    else:
        pred_mask = anomaly_map > score_threshold

    return {
        "engine": model_kind,
        "anomaly_score": round(score, 4),
        "decision_threshold": score_threshold,
        "is_defective": is_defective,
        "detection_confidence": padim_detection_confidence(score, score_threshold, is_defective),
        "anomaly_map": anomaly_map,
        "pred_mask": pred_mask,
        "fallback_used": False,
        "fallback_reason": None,
    }


def predict_with_padim(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    score_threshold: float = 0.5,
    accelerator: str = "auto",
) -> dict:
    """Backward-compatible PaDiM-only entry point."""
    return predict_with_anomaly_model(
        image_path,
        checkpoint_path,
        model_kind="padim",
        score_threshold=score_threshold,
        accelerator=accelerator,
    )
