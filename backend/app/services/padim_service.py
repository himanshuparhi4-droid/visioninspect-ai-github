from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import settings


class PadimInferenceError(RuntimeError):
    pass


def choose_accelerator() -> str:
    configured = settings.padim_inference_accelerator.lower()
    if configured != "auto":
        return configured

    try:
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=1)
def load_padim_runtime(checkpoint_path: str) -> tuple[object, object]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise PadimInferenceError(f"PaDiM checkpoint not found: {path}")

    try:
        from anomalib.engine import Engine
        from anomalib.models import Padim
    except Exception as exc:
        raise PadimInferenceError("Anomalib is not available for PaDiM inference") from exc

    try:
        model = Padim.load_from_checkpoint(path)
        engine = Engine(
            accelerator=choose_accelerator(),
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )
    except Exception as exc:
        raise PadimInferenceError("Could not load PaDiM checkpoint") from exc

    return model, engine


def tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def padim_detection_confidence(score: float, is_defective: bool) -> float:
    from app.services.model_settings_service import load_runtime_settings

    threshold = load_runtime_settings().padim_score_threshold
    margin = abs(score - threshold)
    confidence = 0.58 + min(0.40, margin * 2.0)
    if is_defective and score >= 0.80:
        confidence = max(confidence, 0.90)
    return round(float(max(0.55, min(0.99, confidence))), 4)


def predict_with_padim(image_path: str | Path, checkpoint_path: str | Path) -> dict:
    model, engine = load_padim_runtime(str(checkpoint_path))

    try:
        predictions = engine.predict(
            model=model,
            data_path=Path(image_path),
            return_predictions=True,
        )
    except Exception as exc:
        raise PadimInferenceError("PaDiM prediction failed") from exc

    if not predictions:
        raise PadimInferenceError("PaDiM returned no predictions")

    prediction = predictions[0]
    score = float(tensor_to_numpy(prediction.pred_score).reshape(-1)[0])
    from app.services.model_settings_service import load_runtime_settings

    threshold = load_runtime_settings().padim_score_threshold
    is_defective = score > threshold
    anomaly_map = tensor_to_numpy(prediction.anomaly_map).squeeze().astype(np.float32)

    if hasattr(prediction, "pred_mask") and prediction.pred_mask is not None:
        pred_mask = tensor_to_numpy(prediction.pred_mask).squeeze().astype(bool)
    else:
        pred_mask = anomaly_map > 0.5

    return {
        "engine": "padim",
        "anomaly_score": round(score, 4),
        "decision_threshold": threshold,
        "is_defective": is_defective,
        "detection_confidence": padim_detection_confidence(score, is_defective),
        "anomaly_map": anomaly_map,
        "pred_mask": pred_mask,
    }
