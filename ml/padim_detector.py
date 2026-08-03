from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np


class PadimInferenceError(RuntimeError):
    pass


ANOMALY_MODEL_CACHE_SIZE = max(1, min(int(os.getenv("ANOMALY_MODEL_CACHE_SIZE", "2")), 4))


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
        from anomalib.deploy import OpenVINOInferencer

        return OpenVINOInferencer(
            path=str(path),
            device=device,
        )
    except Exception as exc:
        raise PadimInferenceError(f"Could not load OpenVINO model on {device}") from exc


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

    predictions = None
    if openvino_path and Path(openvino_path).exists():
        try:
            inferencer = load_openvino_runtime(str(openvino_path), openvino_device)

            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))

            ov_preds = inferencer.predict(image=image)
            if ov_preds is not None:
                predictions = [ov_preds]
        except Exception as exc:
            if not (checkpoint_path and Path(checkpoint_path).exists()):
                raise PadimInferenceError("OpenVINO prediction failed") from exc

    if predictions is None:
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
