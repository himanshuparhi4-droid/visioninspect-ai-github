"""Compare category-specific anomaly-detection improvements without keeping temporary checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.anomalib_trainer import build_anomalib_model, build_engine, build_mvtec_datamodule
from ml.config import MVTEC_DATASET_ROOT
from ml.model_registry import category_model_spec

DEFAULT_CATEGORIES = ("cable", "grid", "hazelnut", "screw")
RANDOM_SEED = 42


def seed_everything() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


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


def run_candidate(
    category: str,
    dataset_root: Path,
    model_name: str,
    roi_scale: float,
    *,
    candidate_name: str | None = None,
    image_size: int = 256,
    backbone: str | None = None,
    coreset_sampling_ratio: float = 0.05,
    num_neighbors: int = 5,
) -> dict:
    seed_everything()
    name = candidate_name or model_name
    run_dir = PROJECT_ROOT / "tmp" / "model_benchmarks" / category / name
    shutil.rmtree(run_dir, ignore_errors=True)
    try:
        data = build_mvtec_datamodule(
            root=dataset_root,
            category=category,
            image_size=(image_size, image_size),
            train_batch_size=2 if image_size > 256 or backbone == "wide_resnet50_2" else 4,
            eval_batch_size=2 if image_size > 256 or backbone == "wide_resnet50_2" else 4,
            num_workers=0,
            apply_resize_augmentation=False,
        )
        model = build_anomalib_model(
            model_name,
            image_size=(image_size, image_size),
            roi_scale=roi_scale,
            backbone=backbone,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
        engine = build_engine(run_dir, accelerator="gpu" if torch.cuda.is_available() else "cpu", logger=False)
        engine.fit(model=model, datamodule=data)
        return metric_row(
            name,
            serializable(engine.test(model=model, datamodule=data)),
            model_kind=model_name,
            roi_scale=roi_scale,
            image_size=image_size,
            backbone=backbone or "resnet18",
            coreset_sampling_ratio=coreset_sampling_ratio if model_name == "patchcore" else None,
            num_neighbors=num_neighbors if model_name == "patchcore" else None,
        )
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


def benchmark_category(
    category: str,
    dataset_root: Path,
    strong: bool = False,
    strong_only: bool = False,
    strong_image_size: int = 256,
    strong_coreset_ratio: float = 0.05,
    strong_neighbors: int = 9,
) -> dict:
    spec = category_model_spec(category)
    if not spec.is_trained:
        raise RuntimeError(f"{category} must have a trained PaDiM artifact before benchmarking")
    comparison = {"baseline": baseline_row(category)}
    if not strong_only:
        comparison.update(
            {
                "padim_center_roi": run_candidate(category, dataset_root, "padim", roi_scale=1.15),
                "patchcore": run_candidate(category, dataset_root, "patchcore", roi_scale=1.0),
            }
        )
    if strong or strong_only:
        candidate_name = f"patchcore_wide_{strong_image_size}"
        comparison[candidate_name] = run_candidate(
            category,
            dataset_root,
            "patchcore",
            roi_scale=1.0,
            candidate_name=candidate_name,
            image_size=strong_image_size,
            backbone="wide_resnet50_2",
            coreset_sampling_ratio=strong_coreset_ratio,
            num_neighbors=strong_neighbors,
        )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--dataset-root", type=Path, default=MVTEC_DATASET_ROOT)
    parser.add_argument(
        "--strong",
        action="store_true",
        help="Also benchmark the higher-capacity WideResNet50-2 PatchCore model.",
    )
    parser.add_argument("--strong-image-size", type=int, default=256)
    parser.add_argument("--strong-coreset-ratio", type=float, default=0.05)
    parser.add_argument("--strong-neighbors", type=int, default=9)
    parser.add_argument(
        "--strong-only",
        action="store_true",
        help="Benchmark only the current model and WideResNet50-2 PatchCore candidate.",
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    categories = [item.strip().lower().replace("-", "_") for item in args.categories.split(",") if item.strip()]

    for category in categories:
        print(f"Benchmarking {category}...", flush=True)
        comparison = benchmark_category(
            category,
            dataset_root,
            strong=args.strong,
            strong_only=args.strong_only,
            strong_image_size=args.strong_image_size,
            strong_coreset_ratio=args.strong_coreset_ratio,
            strong_neighbors=args.strong_neighbors,
        )
        spec = category_model_spec(category)
        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
        metadata["benchmark_comparison"] = comparison
        spec.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({category: comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
