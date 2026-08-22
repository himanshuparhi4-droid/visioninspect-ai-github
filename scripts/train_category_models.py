"""Train category-specific PaDiM checkpoints for the complete MVTec AD dataset."""

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
from ml.baseline_detector import (
    build_embedding_bank,
    build_reference_profile,
    build_spatial_embedding_bank,
    save_reference_profile,
)
from ml.config import MODELS_DIR, MVTEC_DATASET_ROOT
from ml.model_registry import SUPPORTED_CATEGORIES, category_model_spec, registry_file


def parse_categories(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(SUPPORTED_CATEGORIES)
    categories = [item.strip().lower().replace("-", "_") for item in value.split(",") if item.strip()]
    unsupported = sorted(set(categories) - set(SUPPORTED_CATEGORIES))
    if unsupported:
        raise ValueError(f"Unsupported categories: {', '.join(unsupported)}")
    return categories


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


def train_category(
    category: str,
    dataset_root: Path,
    model_name: str,
    image_size: int,
    batch_size: int,
    padim_features: int,
    coreset_sampling_ratio: float,
    num_neighbors: int,
    roi_scale: float,
    evaluate: bool,
    keep_training_artifacts: bool,
    candidate: bool,
) -> dict:
    category_root = dataset_root / category
    train_good_dir = category_root / "train" / "good"
    train_images = sorted(train_good_dir.glob("*.png"))
    if not train_images:
        raise FileNotFoundError(f"No normal training images found for category '{category}': {train_good_dir}")

    spec = category_model_spec(category)
    artifact_dir = spec.checkpoint_path.parent
    checkpoint_path = (
        artifact_dir / f"{model_name}_candidate.ckpt" if candidate else spec.checkpoint_path
    )
    metadata_path = (
        artifact_dir / f"{model_name}_candidate_metadata.json" if candidate else spec.metadata_path
    )
    training_dir = artifact_dir / (f"{model_name}_candidate_run" if candidate else "training_runs")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # A category run must not reuse a checkpoint from a previous interrupted run.
    # Only this disposable Anomalib run directory is cleared; final artifacts remain.
    if training_dir.exists():
        shutil.rmtree(training_dir)

    data = build_mvtec_datamodule(
        root=dataset_root,
        category=category,
        image_size=(image_size, image_size),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
    )
    model = build_anomalib_model(
        model_name,
        padim_n_features=padim_features,
        image_size=(image_size, image_size),
        roi_scale=roi_scale,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
    )
    engine = build_engine(training_dir, accelerator="gpu" if torch.cuda.is_available() else "cpu", logger=False)
    engine.fit(model=model, datamodule=data)

    source_checkpoint = latest_checkpoint(training_dir)
    shutil.copy2(source_checkpoint, checkpoint_path)

    if not candidate:
        profile = build_reference_profile(train_images, size=(image_size, image_size))
        profile["embedding_bank"] = build_embedding_bank(train_images)
        profile["spatial_embedding_bank"] = build_spatial_embedding_bank(train_images)
        save_reference_profile(profile, spec.baseline_profile_path)

    metrics = engine.test(model=model, datamodule=data) if evaluate else {}
    existing_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata_path.exists()
        else {}
    )
    model_history = list(existing_metadata.get("model_history", []))
    if existing_metadata.get("metrics"):
        model_history.append(
            {
                "model_name": existing_metadata.get("model_name"),
                "model_version": existing_metadata.get("model_version"),
                "padim_features": existing_metadata.get("padim_features"),
                "trained_at": existing_metadata.get("trained_at"),
                "metrics": existing_metadata.get("metrics"),
            }
        )
        model_history = model_history[-3:]
    metadata = {
        **existing_metadata,
        "project": "VisionInspect AI",
        "model_name": model_name,
        "model_version": "candidate" if candidate else "v1",
        "category": category,
        "dataset": "MVTec AD",
        "image_size": [image_size, image_size],
        "input_quality": {
            "resolution": image_size,
            "roi_scale": roi_scale,
            "preprocessor": "resize" if roi_scale == 1.0 else "resize_then_center_crop_roi",
        },
        "padim_features": padim_features,
        "coreset_sampling_ratio": coreset_sampling_ratio if model_name == "patchcore" else None,
        "num_neighbors": num_neighbors if model_name == "patchcore" else None,
        "train_good_images": len(train_images),
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "checkpoint_path": str(checkpoint_path),
        "baseline_profile_path": str(spec.baseline_profile_path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": serializable_metrics(metrics),
        "model_history": model_history,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not keep_training_artifacts:
        shutil.rmtree(training_dir, ignore_errors=True)
    return metadata


def build_portable_category_profile(category: str, dataset_root: Path, image_size: int) -> dict:
    """Build only the compact normal profile used by a regular Git clone."""
    spec = category_model_spec(category)
    train_images = sorted((dataset_root / category / "train" / "good").glob("*.png"))
    if not train_images:
        raise FileNotFoundError(f"No normal training images found for category '{category}'.")

    profile = build_reference_profile(train_images, size=(image_size, image_size))
    profile["embedding_bank"] = build_embedding_bank(train_images)
    profile["spatial_embedding_bank"] = build_spatial_embedding_bank(train_images)
    save_reference_profile(profile, spec.baseline_profile_path)

    metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8")) if spec.metadata_path.exists() else {}
    metadata.update(
        {
            "project": "VisionInspect AI",
            "category": category,
            "dataset": "MVTec AD",
            "portable_detector": "resnet18_normal_memory_with_opencv_localization",
            "portable_train_good_images": len(train_images),
            "baseline_profile_path": str(spec.baseline_profile_path),
        }
    )
    spec.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def promote_candidate(category: str, model_name: str) -> dict:
    """Promote a validated candidate without discarding category metadata."""
    spec = category_model_spec(category)
    artifact_dir = spec.checkpoint_path.parent
    candidate_checkpoint = artifact_dir / f"{model_name}_candidate.ckpt"
    candidate_metadata_path = artifact_dir / f"{model_name}_candidate_metadata.json"
    if not candidate_checkpoint.exists() or not candidate_metadata_path.exists():
        raise FileNotFoundError(f"No {model_name} candidate is available for '{category}'.")

    active_metadata = (
        json.loads(spec.metadata_path.read_text(encoding="utf-8-sig"))
        if spec.metadata_path.exists()
        else {}
    )
    candidate_metadata = json.loads(candidate_metadata_path.read_text(encoding="utf-8-sig"))
    history = list(active_metadata.get("model_history", []))
    if active_metadata.get("metrics"):
        history.append(
            {
                "model_name": active_metadata.get("model_name"),
                "model_version": active_metadata.get("model_version"),
                "padim_features": active_metadata.get("padim_features"),
                "coreset_sampling_ratio": active_metadata.get("coreset_sampling_ratio"),
                "trained_at": active_metadata.get("trained_at"),
                "metrics": active_metadata.get("metrics"),
            }
        )

    destination = artifact_dir / f"{model_name}_v1.ckpt"
    shutil.copy2(candidate_checkpoint, destination)
    promoted_metadata = {
        **active_metadata,
        **candidate_metadata,
        "model_name": model_name,
        "model_version": "v1",
        "checkpoint_path": str(destination),
        "model_history": history[-4:],
    }
    spec.metadata_path.write_text(json.dumps(promoted_metadata, indent=2), encoding="utf-8")

    path = registry_file()
    registry = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    entry = registry.setdefault(category, {})
    entry["model_kind"] = model_name
    entry["checkpoint_path"] = str(destination.relative_to(MODELS_DIR.parent)).replace("\\", "/")
    entry["input_size"] = int(candidate_metadata.get("image_size", [256])[0])
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    candidate_checkpoint.unlink()
    candidate_metadata_path.unlink()
    return promoted_metadata


def update_registry(categories: list[str]) -> None:
    path = registry_file()
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for category in categories:
        spec = category_model_spec(category)
        entry = existing.setdefault(category, {})
        entry.update(
            {
                "model_kind": spec.model_kind,
                "checkpoint_path": str(spec.checkpoint_path.relative_to(MODELS_DIR.parent)).replace("\\", "/"),
                "classifier_path": str(spec.classifier_path.relative_to(MODELS_DIR.parent)).replace("\\", "/"),
                "baseline_profile_path": str(spec.baseline_profile_path.relative_to(MODELS_DIR.parent)).replace(
                    "\\", "/"
                ),
                "metadata_path": str(spec.metadata_path.relative_to(MODELS_DIR.parent)).replace("\\", "/"),
                "padim_score_threshold": spec.padim_score_threshold,
                "baseline_score_threshold": spec.baseline_score_threshold,
                "baseline_residual_threshold": spec.baseline_residual_threshold,
            }
        )
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default="all", help="Comma-separated MVTec categories, or 'all'.")
    parser.add_argument("--dataset-root", type=Path, default=MVTEC_DATASET_ROOT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-name", choices=("padim", "patchcore"), default="padim")
    parser.add_argument(
        "--padim-features",
        type=int,
        default=50,
        help="Compact PaDiM feature count. Fifty keeps OpenVINO exports practical for CPU deployment.",
    )
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.05)
    parser.add_argument("--num-neighbors", type=int, default=5)
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Write a comparison checkpoint without replacing the active category runtime.",
    )
    parser.add_argument(
        "--promote-candidate",
        action="store_true",
        help="Promote an already evaluated candidate and update the category registry.",
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=1.0,
        help="Resize larger then center-crop to image-size. Use 1.10-1.25 for tighter object focus.",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--keep-training-artifacts", action="store_true")
    parser.add_argument(
        "--portable-only",
        action="store_true",
        help="Rebuild compact OpenCV/ResNet profiles without retraining PaDiM or PatchCore.",
    )
    args = parser.parse_args()

    categories = parse_categories(args.categories)
    if args.promote_candidate:
        for category in categories:
            metadata = promote_candidate(category, args.model_name)
            print(f"Promoted {category}: {metadata['checkpoint_path']}", flush=True)
        return

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    update_registry(categories)
    for index, category in enumerate(categories, start=1):
        if args.portable_only:
            print(f"[{index}/{len(categories)}] Building portable profile for {category}...", flush=True)
            metadata = build_portable_category_profile(category, dataset_root, args.image_size)
            print(f"Completed {category}: {metadata['baseline_profile_path']}", flush=True)
            continue
        print(f"[{index}/{len(categories)}] Training {args.model_name} for {category}...", flush=True)
        metadata = train_category(
            category,
            dataset_root,
            args.model_name,
            args.image_size,
            args.batch_size,
            args.padim_features,
            args.coreset_sampling_ratio,
            args.num_neighbors,
            args.roi_scale,
            not args.skip_evaluation,
            args.keep_training_artifacts,
            args.candidate,
        )
        print(f"Completed {category}: {metadata['checkpoint_path']}", flush=True)


if __name__ == "__main__":
    main()
