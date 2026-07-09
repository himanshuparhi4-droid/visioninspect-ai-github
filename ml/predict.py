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
    """Run the same backend-ready inspection pipeline used by the app."""
    from app.services.prediction_service import inspect_image_file

    selected_image = Path(image_path) if image_path else default_demo_image()
    result = inspect_image_file(selected_image)
    return {
        "input_image": str(selected_image),
        "prediction": result["prediction"],
        "defect_type": result["defect_type"],
        "confidence": result["confidence"],
        "anomaly_score": result["anomaly_score"],
        "defect_area_ratio": result["defect_area_ratio"],
        "heatmap_path": result["heatmap_path"],
        "processed_image_path": result["processed_image_path"],
        "severity_score": result["severity_score"],
        "severity_level": result["severity_level"],
        "pass_fail": result["pass_fail"],
        "recommended_action": result["recommended_action"],
        "model_used": result["model_version"],
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
