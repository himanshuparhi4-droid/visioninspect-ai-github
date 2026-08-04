"""Train category-specific experimental defect classifiers from labelled MVTec image folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.classifier import (
    GLOBAL_TEXTURE_FEATURE_MODE,
    HANDCRAFTED_ROI_SHAPE_FEATURE_MODE,
    ROI_PIXEL_TEXTURE_FEATURE_MODE,
    ROI_SHAPE_TEXTURE_FEATURE_MODE,
    ROI_TEXTURE_FEATURE_MODE,
    export_portable_forest,
    train_defect_classifier,
)
from ml.config import MVTEC_DATASET_ROOT
from ml.model_registry import SUPPORTED_CATEGORIES, category_model_spec, registry_file
from ml.padim_detector import load_openvino_runtime


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


def attach_openvino_masks(records: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"OpenVINO model not found: {model_path}")
    compiled_model = load_openvino_runtime(str(model_path), "CPU")
    updated = records.copy()
    masks = []
    for image_path in updated["image_path"]:
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
        outputs = compiled_model([image.transpose(2, 0, 1)[None]])
        anomaly_map = np.asarray(outputs[compiled_model.output("anomaly_map")]).squeeze()
        mask = np.asarray(outputs[compiled_model.output("pred_mask")]).squeeze().astype(bool)
        if not np.any(mask):
            mask = anomaly_map >= np.percentile(anomaly_map, 99)
        masks.append(mask)
    updated["mask"] = masks
    return updated


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
            HANDCRAFTED_ROI_SHAPE_FEATURE_MODE,
        ),
    )
    parser.add_argument("--force", action="store_true", help="Promote the best new classifier even if metrics are lower.")
    parser.add_argument(
        "--mask-source",
        choices=("ground_truth", "openvino"),
        default="ground_truth",
        help="Use annotation masks or production-style OpenVINO anomaly masks for ROI features.",
    )
    args = parser.parse_args()

    categories = (
        SUPPORTED_CATEGORIES
        if args.categories == "all"
        else tuple(item.strip().lower().replace("-", "_") for item in args.categories.split(",") if item.strip())
    )
    for category in categories:
        spec = category_model_spec(category)
        records = category_records(args.dataset_root / category)
        if args.mask_source == "openvino":
            if spec.openvino_path is None:
                raise FileNotFoundError(f"No OpenVINO model configured for {category}")
            records = attach_openvino_masks(records, spec.openvino_path)
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
                HANDCRAFTED_ROI_SHAPE_FEATURE_MODE,
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
                        "defect-only subtype classification with stratified cross-validation using "
                        f"{args.mask_source.replace('_', ' ')} masks; "
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
            if spec.compact_classifier_path is not None and best_result["metrics"]["feature_mode"] == HANDCRAFTED_ROI_SHAPE_FEATURE_MODE:
                export_portable_forest(
                    best_result["bundle"]["classifier"],
                    spec.compact_classifier_path,
                    feature_mode=HANDCRAFTED_ROI_SHAPE_FEATURE_MODE,
                )
                registry_path = registry_file()
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                registry.setdefault(category, {})["compact_classifier_path"] = str(
                    spec.compact_classifier_path.relative_to(PROJECT_ROOT)
                ).replace("\\", "/")
                registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
            metadata["defect_classifier"] = best_result["metrics"]
            spec.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
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
