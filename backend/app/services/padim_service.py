from __future__ import annotations

import sys
from pathlib import Path

from app.config import settings
from app.services.model_settings_service import load_runtime_settings

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.padim_detector import PadimInferenceError
from ml.padim_detector import choose_accelerator as _choose_accelerator
from ml.padim_detector import load_padim_runtime as _load_padim_runtime
from ml.padim_detector import predict_with_padim as _predict_with_padim
from ml.padim_detector import run_quietly, tensor_to_numpy


def choose_accelerator() -> str:
    return _choose_accelerator(settings.padim_inference_accelerator)


def load_padim_runtime(checkpoint_path: str) -> tuple[object, object]:
    return _load_padim_runtime(checkpoint_path, settings.padim_inference_accelerator)


def predict_with_padim(image_path: str | Path, checkpoint_path: str | Path) -> dict:
    runtime_settings = load_runtime_settings()
    return _predict_with_padim(
        image_path,
        checkpoint_path,
        score_threshold=runtime_settings.padim_score_threshold,
        accelerator=settings.padim_inference_accelerator,
    )


__all__ = [
    "PadimInferenceError",
    "choose_accelerator",
    "load_padim_runtime",
    "predict_with_padim",
    "run_quietly",
    "tensor_to_numpy",
]
