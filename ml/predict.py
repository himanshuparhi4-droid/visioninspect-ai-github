from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def default_demo_image() -> Path:
    """Return a real MVTec bottle image for command-line demonstration."""
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection" / "bottle" / "test" / "contamination",
        PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection" / "bottle" / "test" / "broken_large",
        PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection" / "bottle" / "test" / "broken_small",
        PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection" / "bottle" / "test" / "good",
    ]
    for directory in candidates:
        image_paths = sorted(directory.glob("*.png"))
        if image_paths:
            return image_paths[0]
    raise FileNotFoundError("No MVTec bottle test image found under data/raw.")


def inspect_image(image_path: str | Path | None = None) -> dict:
    """Run the backend-compatible AI pipeline without creating external output files."""
    from app.config import settings
    from app.services.model_settings_service import load_runtime_settings
    from app.services.prediction_service import resolve_backend_path
    from ml.inference import InferenceConfig, inspect_image as inspect_image_runtime

    selected_image = Path(image_path) if image_path else default_demo_image()
    runtime_settings = load_runtime_settings()
    config = InferenceConfig(
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
    result = inspect_image_runtime(selected_image, config)
    return {
        "input_image": str(selected_image),
        "prediction": result["prediction"],
        "defect_type": result["defect_type"],
        "confidence": result["confidence"],
        "anomaly_score": result["anomaly_score"],
        "defect_area_ratio": result["defect_area_ratio"],
        "heatmap_path": "inline:not_saved_by_cli",
        "processed_image_path": "inline:not_saved_by_cli",
        "severity_score": result["severity_score"],
        "severity_level": result["severity_level"],
        "pass_fail": result["pass_fail"],
        "recommended_action": result["recommended_action"],
        "model_used": result["model_used"],
        "active_inference_engine": result["active_inference_engine"],
        "fallback_used": result["fallback_used"],
        "fallback_reason": result["fallback_reason"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VisionInspect AI inference on one image.")
    parser.add_argument("--image", type=str, default=None, help="Path to an input product image.")
    parser.add_argument(
        "--use-padim",
        action="store_true",
        help="Force PaDiM checkpoint inference for this run.",
    )
    parser.add_argument(
        "--use-baseline",
        action="store_true",
        help="Force OpenCV baseline inference for this run.",
    )
    args = parser.parse_args()

    if args.use_padim and args.use_baseline:
        raise SystemExit("Choose only one: --use-padim or --use-baseline.")
    if args.use_padim:
        os.environ["USE_PADIM_INFERENCE"] = "true"
    if args.use_baseline:
        os.environ["USE_PADIM_INFERENCE"] = "false"

    print(json.dumps(inspect_image(args.image), indent=2))


if __name__ == "__main__":
    main()
