"""Train category-specific experimental defect classifiers from labelled MVTec image folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.classifier import (
    GLOBAL_TEXTURE_FEATURE_MODE,
    ROI_PIXEL_TEXTURE_FEATURE_MODE,
    ROI_SHAPE_TEXTURE_FEATURE_MODE,
    ROI_TEXTURE_FEATURE_MODE,
    train_defect_classifier,
)
from ml.config import MVTEC_DATASET_ROOT
from ml.model_registry import SUPPORTED_CATEGORIES, category_model_spec


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


def remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default="all")
    parser.add_argument("--dataset-root", type=Path, default=MVTEC_DATASET_ROOT)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument(
        "--feature-mode",
        default="auto",
        choices=(
            "auto",
            GLOBAL_TEXTURE_FEATURE_MODE,
            ROI_TEXTURE_FEATURE_MODE,
            ROI_SHAPE_TEXTURE_FEATURE_MODE,
            ROI_PIXEL_TEXTURE_FEATURE_MODE,
        ),
    )
    parser.add_argument("--force", action="store_true", help="Promote the best new classifier even if metrics are lower.")
    args = parser.parse_args()

    categories = (
        SUPPORTED_CATEGORIES
        if args.categories == "all"
        else tuple(item.strip().lower().replace("-", "_") for item in args.categories.split(",") if item.strip())
    )
    for category in categories:
        spec = category_model_spec(category)
        records = category_records(args.dataset_root / category)
        defect_records = records[records["label"] != "good"]
        counts = defect_records["label"].value_counts()
        if defect_records.empty or (counts < 2).any():
            print(f"Skipping {category}: insufficient labelled images for stratified classifier split")
            continue
        print(
            f"Training defect-subtype classifier for {category} "
            f"({len(defect_records)} defective images, {len(counts)} subtypes)...",
            flush=True,
        )
        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
        current_metrics = metadata.get("defect_classifier", {})
        feature_modes = (
            (
                GLOBAL_TEXTURE_FEATURE_MODE,
                ROI_TEXTURE_FEATURE_MODE,
                ROI_SHAPE_TEXTURE_FEATURE_MODE,
                ROI_PIXEL_TEXTURE_FEATURE_MODE,
            )
            if args.feature_mode == "auto"
            else (args.feature_mode,)
        )
        best_result = None
        best_path = None
        candidate_paths = []
        for feature_mode in feature_modes:
            candidate_path = spec.classifier_path.with_name(
                f"{spec.classifier_path.stem}.{feature_mode}.candidate.pkl"
            )
            candidate_paths.append(candidate_path)
            result = train_defect_classifier(
                records,
                candidate_path,
                test_size=args.test_size,
                label_order=sorted(counts.index),
                dataset_context={
                    "source": "MVTec AD labelled test folders",
                    "protocol": (
                        "defect-only subtype classification with stratified cross-validation; "
                        "not an official held-out MVTec anomaly-detection benchmark"
                    ),
                    "category": category,
                },
                defect_only=True,
                feature_mode=feature_mode,
                cross_validate_model=True,
            )
            metrics = result["metrics"]
            print(
                f"  {category}/{feature_mode}: accuracy={metrics['accuracy']}, "
                f"macro_f1={metrics['macro_f1']}, "
                f"selected={metrics['evaluation']['selected_classifier']}",
                flush=True,
            )
            if best_result is None or is_better(metrics, best_result["metrics"]):
                best_result = result
                best_path = candidate_path

        assert best_result is not None and best_path is not None
        should_promote = args.force or not current_metrics or is_better(best_result["metrics"], current_metrics)
        if should_promote:
            best_path.replace(spec.classifier_path)
            metadata["defect_classifier"] = best_result["metrics"]
            spec.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            status = "promoted"
        else:
            status = "kept-existing"

        for candidate_path in candidate_paths:
            if candidate_path != best_path or not should_promote:
                remove_file(candidate_path)

        metrics = best_result["metrics"]
        print(
            f"Completed {category}: {status}, accuracy={metrics['accuracy']}, "
            f"macro_f1={metrics['macro_f1']}, feature_mode={metrics['feature_mode']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
