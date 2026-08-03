"""Compare category-specific anomaly-detection improvements without keeping temporary checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.anomalib_trainer import build_anomalib_model, build_engine, build_mvtec_datamodule
from ml.config import MVTEC_DATASET_ROOT
from ml.model_registry import category_model_spec

DEFAULT_CATEGORIES = ("cable", "grid", "hazelnut", "screw")


def serializable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def metric_row(name: str, metrics: object, **details: object) -> dict:
    values = metrics[0] if isinstance(metrics, list) and metrics else metrics
    values = values if isinstance(values, dict) else {}
    return {
        "name": name,
        "image_auroc": values.get("image_AUROC"),
        "image_f1": values.get("image_F1Score"),
        "pixel_auroc": values.get("pixel_AUROC"),
        "pixel_f1": values.get("pixel_F1Score"),
        **details,
    }


def run_candidate(category: str, dataset_root: Path, model_name: str, roi_scale: float) -> dict:
    run_dir = PROJECT_ROOT / "tmp" / "model_benchmarks" / category / f"{model_name}_roi_{roi_scale}"
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
        model = build_anomalib_model(model_name, image_size=(256, 256), roi_scale=roi_scale)
        engine = build_engine(run_dir, accelerator="gpu" if torch.cuda.is_available() else "cpu", logger=False)
        engine.fit(model=model, datamodule=data)
        return metric_row(model_name, serializable(engine.test(model=model, datamodule=data)), roi_scale=roi_scale)
    except Exception as exc:
        return {"name": model_name, "roi_scale": roi_scale, "error": str(exc)}
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def baseline_row(category: str) -> dict:
    spec = category_model_spec(category)
    metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
    return metric_row("padim_current", metadata.get("metrics", {}), roi_scale=1.0)


def benchmark_category(category: str, dataset_root: Path) -> dict:
    spec = category_model_spec(category)
    if not spec.is_trained:
        raise RuntimeError(f"{category} must have a trained PaDiM artifact before benchmarking")
    return {
        "baseline": baseline_row(category),
        "padim_center_roi": run_candidate(category, dataset_root, "padim", roi_scale=1.15),
        "patchcore": run_candidate(category, dataset_root, "patchcore", roi_scale=1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--dataset-root", type=Path, default=MVTEC_DATASET_ROOT)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    categories = [item.strip().lower().replace("-", "_") for item in args.categories.split(",") if item.strip()]

    for category in categories:
        print(f"Benchmarking {category}...", flush=True)
        comparison = benchmark_category(category, dataset_root)
        spec = category_model_spec(category)
        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
        metadata["benchmark_comparison"] = comparison
        spec.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({category: comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
