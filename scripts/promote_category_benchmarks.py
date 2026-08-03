"""Retrain and promote measured category-model winners into the live registry."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.anomalib_trainer import build_anomalib_model, build_engine, build_mvtec_datamodule
from ml.config import MODELS_DIR, MVTEC_DATASET_ROOT
from ml.model_registry import category_model_spec, registry_file


def serializable_metrics(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): serializable_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable_metrics(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def latest_checkpoint(training_dir: Path) -> Path:
    checkpoints = sorted(training_dir.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError(f"Anomalib did not produce a checkpoint in {training_dir}")
    return checkpoints[-1]

WINNERS = {
    "cable": {"model_kind": "patchcore", "roi_scale": 1.0},
    "grid": {"model_kind": "padim", "roi_scale": 1.15},
    "hazelnut": {"model_kind": "patchcore", "roi_scale": 1.0},
    "screw": {"model_kind": "patchcore", "roi_scale": 1.0},
}


def promoted_checkpoint_path(category: str, model_kind: str) -> Path:
    return MODELS_DIR / "categories" / category / f"{model_kind}_v1.ckpt"


def promote(category: str, dataset_root: Path) -> dict:
    winner = WINNERS[category]
    model_kind = winner["model_kind"]
    roi_scale = winner["roi_scale"]
    spec = category_model_spec(category)
    train_dir = dataset_root / category / "train" / "good"
    if not any(train_dir.glob("*.png")):
        raise FileNotFoundError(f"Missing normal training images for '{category}': {train_dir}")
    artifact_dir = spec.metadata_path.parent
    run_dir = artifact_dir / "promotion_runs"
    shutil.rmtree(run_dir, ignore_errors=True)
    try:
        data = build_mvtec_datamodule(
            root=dataset_root,
            category=category,
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            apply_resize_augmentation=False,
        )
        model = build_anomalib_model(model_kind, image_size=(256, 256), roi_scale=roi_scale)
        engine = build_engine(run_dir, accelerator="gpu" if torch.cuda.is_available() else "cpu", logger=False)
        engine.fit(model=model, datamodule=data)
        destination = promoted_checkpoint_path(category, model_kind)
        shutil.copy2(latest_checkpoint(run_dir), destination)
        metrics = serializable_metrics(engine.test(model=model, datamodule=data))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "model_name": model_kind,
            "model_version": "v1",
            "checkpoint_path": str(destination),
            "roi_scale": roi_scale,
            "selected_from_benchmark": True,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
    )
    metadata.pop("threshold_calibration", None)
    spec.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"checkpoint_path": destination, "model_kind": model_kind, "roi_scale": roi_scale}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories",
        default=",".join(WINNERS),
        help="Comma-separated benchmark winners to promote.",
    )
    parser.add_argument("--dataset-root", type=Path, default=MVTEC_DATASET_ROOT)
    args = parser.parse_args()
    categories = [item.strip().lower().replace("-", "_") for item in args.categories.split(",") if item.strip()]
    unknown = sorted(set(categories) - set(WINNERS))
    if unknown:
        raise ValueError(f"No benchmark promotion is defined for: {', '.join(unknown)}")
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    registry_path = registry_file()
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    for category in categories:
        print(f"Promoting {category}...", flush=True)
        result = promote(category, dataset_root)
        entry = registry.setdefault(category, {})
        entry["model_kind"] = result["model_kind"]
        entry["checkpoint_path"] = str(result["checkpoint_path"].relative_to(MODELS_DIR.parent)).replace("\\", "/")
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        print(f"Promoted {category} with {result['model_kind']}", flush=True)


if __name__ == "__main__":
    main()
