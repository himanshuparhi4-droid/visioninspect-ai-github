"""Train optional fine-tuned CNN defect subtype classifiers for weak categories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.cnn_classifier import (
    CNN_CANDIDATE_FILE,
    CNN_ONNX_FILE,
    export_cnn_classifier_onnx,
    train_cnn_defect_classifier,
)
from ml.config import MVTEC_DATASET_ROOT
from ml.model_registry import SUPPORTED_CATEGORIES, category_model_spec, registry_file
from scripts.train_category_classifiers import attach_runtime_masks

DEFAULT_CATEGORIES = ("capsule", "grid", "wood")


def parse_categories(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return SUPPORTED_CATEGORIES
    categories = tuple(item.strip().lower().replace("-", "_") for item in value.split(",") if item.strip())
    unsupported = sorted(set(categories) - set(SUPPORTED_CATEGORIES))
    if unsupported:
        raise ValueError(f"Unsupported categories: {', '.join(unsupported)}")
    return categories


def category_records(category_root: Path) -> pd.DataFrame:
    records = []
    test_root = category_root / "test"
    for label_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        for image_path in sorted(label_dir.glob("*.png")):
            mask_path = category_root / "ground_truth" / label_dir.name / f"{image_path.stem}_mask.png"
            records.append(
                {
                    "label": label_dir.name,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path.exists() else None,
                }
            )
    return pd.DataFrame(records)


def metric_floor(metrics: dict) -> float:
    return min(float(metrics.get("accuracy", 0.0)), float(metrics.get("macro_f1", 0.0)))


def is_better(candidate: dict, current: dict) -> bool:
    candidate_floor = metric_floor(candidate)
    current_floor = metric_floor(current)
    if candidate_floor > current_floor:
        return True
    if candidate_floor == current_floor:
        return float(candidate.get("macro_f1", 0.0)) > float(current.get("macro_f1", 0.0))
    return False


def update_registry(category: str, cnn_path: Path) -> None:
    path = registry_file()
    registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    entry = registry.setdefault(category, {})
    entry["cnn_classifier_path"] = str(cnn_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--dataset-root", type=Path, default=MVTEC_DATASET_ROOT)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--crop-mode",
        choices=("auto", "bbox", "defect", "object", "full"),
        default="auto",
        help="CNN input view. auto compares tight bbox, defect, object, and full views.",
    )
    parser.add_argument(
        "--mask-source",
        choices=("ground_truth", "openvino", "render"),
        default="openvino",
        help="Localization masks used for CNN crops. OpenVINO is the intended Render deployment pair.",
    )
    parser.add_argument("--force", action="store_true", help="Promote CNN even if current classifier metrics are better.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    for category in parse_categories(args.categories):
        spec = category_model_spec(category)
        records = category_records(dataset_root / category)
        if args.mask_source in {"openvino", "render"}:
            records = attach_runtime_masks(records, spec, profile=args.mask_source)
        defect_records = records[records["label"] != "good"]
        counts = defect_records["label"].value_counts()
        if defect_records.empty or (counts < 2).any():
            print(f"Skipping {category}: insufficient labelled subtype data", flush=True)
            continue

        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8")) if spec.metadata_path.exists() else {}
        current_metrics = metadata.get("defect_classifier", {})
        final_path = spec.classifier_path.with_name(CNN_ONNX_FILE)
        crop_modes = ("bbox", "defect", "object", "full") if args.crop_mode == "auto" else (args.crop_mode,)
        best_result = None
        best_candidate_path = None
        candidate_paths = []
        print(
            f"Training fine-tuned CNN for {category}: {len(defect_records)} defective images, "
            f"{len(counts)} subtypes, image_size={args.image_size}",
            flush=True,
        )
        for crop_mode in crop_modes:
            candidate_path = spec.classifier_path.with_name(
                f"{Path(CNN_CANDIDATE_FILE).stem}.{crop_mode}.pt"
            )
            candidate_paths.append(candidate_path)
            result = train_cnn_defect_classifier(
                records,
                candidate_path,
                category=category,
                image_size=args.image_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                label_order=sorted(counts.index),
                crop_mode=crop_mode,
            )
            metrics = result["metrics"]
            print(
                f"  CNN/{crop_mode}: accuracy={metrics['accuracy']}, macro_f1={metrics['macro_f1']}, "
                f"best_epoch={metrics['best_epoch']}",
                flush=True,
            )
            if best_result is None or is_better(metrics, best_result["metrics"]):
                best_result = result
                best_candidate_path = candidate_path

        assert best_result is not None and best_candidate_path is not None
        metrics = best_result["metrics"]
        metadata["defect_classifier_revalidation"] = metrics
        print(
            f"  Best CNN result: accuracy={metrics['accuracy']}, macro_f1={metrics['macro_f1']}, "
            f"crop_mode={metrics['dataset_context']['crop_mode']}",
            flush=True,
        )
        if args.force or not current_metrics or is_better(metrics, current_metrics):
            export_cnn_classifier_onnx(best_candidate_path, final_path)
            metadata["defect_classifier"] = metrics
            metadata["defect_classifier"]["active_artifact"] = str(final_path.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            )
            spec.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            update_registry(category, final_path)
            print(f"Completed {category}: promoted CNN classifier", flush=True)
        else:
            print(
                f"Completed {category}: kept existing classifier "
                f"(current accuracy={current_metrics.get('accuracy')}, macro_f1={current_metrics.get('macro_f1')})",
                flush=True,
            )
        spec.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for candidate_path in candidate_paths:
            if candidate_path.exists():
                candidate_path.unlink()


if __name__ == "__main__":
    main()
