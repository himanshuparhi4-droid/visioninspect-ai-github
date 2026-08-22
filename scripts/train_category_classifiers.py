"""Train category-specific experimental defect classifiers from labelled MVTec image folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.classifier import (
    GLOBAL_TEXTURE_FEATURE_MODE,
    HANDCRAFTED_ROI_SHAPE_FEATURE_MODE,
    ROI_PIXEL_TEXTURE_FEATURE_MODE,
    ROI_SHAPE_TEXTURE_FEATURE_MODE,
    ROI_TEXTURE_FEATURE_MODE,
    _extract_training_features,
    export_portable_forest,
    train_defect_classifier,
)
from ml.config import MVTEC_DATASET_ROOT
from ml.defect_classifier import (
    predict_portable_cnn_defect_type,
    refine_defect_mask_for_classification,
    shared_feature_runtime,
)
from ml.model_registry import (
    SUPPORTED_CATEGORIES,
    category_model_spec,
    classifier_runtime_status,
    registry_file,
)
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


def attach_openvino_masks(
    records: pd.DataFrame,
    model_path: Path,
    input_size: int,
    *,
    category: str,
    mask_policy: str = "model",
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"OpenVINO model not found: {model_path}")
    compiled_model = load_openvino_runtime(str(model_path), "CPU")
    updated = records.copy()
    masks = []
    for image_path in updated["image_path"]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size)).astype(np.float32) / 255.0
        outputs = compiled_model([image.transpose(2, 0, 1)[None]])
        anomaly_map = np.asarray(outputs[compiled_model.output("anomaly_map")]).squeeze()
        model_mask = np.asarray(outputs[compiled_model.output("pred_mask")]).squeeze().astype(bool)
        if mask_policy == "model":
            mask = model_mask
        elif mask_policy.startswith("p"):
            percentile = float(mask_policy[1:])
            mask = anomaly_map >= np.percentile(anomaly_map, percentile)
        else:
            raise ValueError(f"Unsupported classification mask policy: {mask_policy}")
        if not np.any(mask):
            mask = anomaly_map >= np.percentile(anomaly_map, 99)
        if category == "capsule":
            mask = refine_defect_mask_for_classification(mask)
        masks.append(mask)
    updated["mask"] = masks
    return updated


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = (confidences >= lower) & (confidences <= upper if upper == 1.0 else confidences < upper)
        if not np.any(selected):
            continue
        error += float(selected.mean()) * abs(float(confidences[selected].mean()) - float(correct[selected].mean()))
    return round(error, 4)


def confidence_calibration(confidences: np.ndarray, correct: np.ndarray) -> dict:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(confidences) < 30 or len(np.unique(confidences)) < 2:
        return {
            "method": "not_available",
            "reason": "Insufficient independent confidence variation",
            "review_threshold": 0.55,
            "target_accuracy": 0.85,
            "target_achieved": bool(correct.mean() >= 0.85),
            "raw_ece": expected_calibration_error(confidences, correct),
        }

    split_count = min(5, max(2, len(confidences) // 10))
    splitter = KFold(n_splits=split_count, shuffle=True, random_state=42)
    cross_fitted = np.empty_like(confidences)
    for train_index, test_index in splitter.split(confidences):
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(confidences[train_index], correct[train_index])
        cross_fitted[test_index] = calibrator.predict(confidences[test_index])

    final_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    final_calibrator.fit(confidences, correct)
    raw_ece = expected_calibration_error(confidences, correct)
    calibrated_ece = expected_calibration_error(cross_fitted, correct)
    calibration_improved = calibrated_ece < raw_ece
    confidence_for_review = cross_fitted if calibration_improved else confidences
    review_threshold = 0.95
    target_achieved = False
    coverage = 0.0
    accepted_accuracy = 0.0
    for threshold in np.linspace(0.5, 0.95, 19):
        accepted = confidence_for_review >= threshold
        candidate_coverage = float(accepted.mean())
        if candidate_coverage < 0.20:
            continue
        candidate_accuracy = float(correct[accepted].mean())
        if candidate_accuracy >= 0.85:
            review_threshold = float(threshold)
            target_achieved = True
            coverage = candidate_coverage
            accepted_accuracy = candidate_accuracy
            break
    if not target_achieved:
        accepted = confidence_for_review >= review_threshold
        coverage = float(accepted.mean())
        accepted_accuracy = float(correct[accepted].mean()) if np.any(accepted) else 0.0

    result = {
        "method": "isotonic_oof_top_label" if calibration_improved else "not_applied_no_ece_gain",
        "review_threshold": round(review_threshold, 4),
        "target_accuracy": 0.85,
        "target_achieved": target_achieved,
        "coverage": round(coverage, 4),
        "accepted_accuracy": round(accepted_accuracy, 4),
        "raw_ece": raw_ece,
        "cross_fitted_calibrated_ece": calibrated_ece,
    }
    if calibration_improved:
        result["x_thresholds"] = [round(float(value), 6) for value in final_calibrator.X_thresholds_]
        result["y_thresholds"] = [round(float(value), 6) for value in final_calibrator.y_thresholds_]
    else:
        result["reason"] = "Cross-fitted isotonic calibration did not improve expected calibration error"
    return result


def evaluated_subtype_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_order: list[str],
    *,
    protocol: str,
    classifier_engine: str,
    feature_mode: str,
) -> dict:
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 4),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=label_order,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=label_order).tolist(),
        "labels": label_order,
        "samples": int(len(labels)),
        "protocol": protocol,
        "classifier_engine": classifier_engine,
        "feature_mode": feature_mode,
        "mask_source": "active OpenVINO anomaly detector",
        "confidence_calibration": confidence_calibration(confidences, correct),
    }


def evaluate_active_runtime(category: str, records: pd.DataFrame, spec) -> dict:
    defect_records = records[records["label"] != "good"].reset_index(drop=True)
    labels = defect_records["label"].to_numpy()
    label_order = sorted(defect_records["label"].unique().tolist())
    runtime = classifier_runtime_status(spec)
    engine = str(runtime["engine"])

    if engine == "fine_tuned_resnet18_onnx" and spec.cnn_classifier_path is not None:
        _, test_index = train_test_split(
            np.arange(len(defect_records)),
            test_size=0.20,
            stratify=labels,
            random_state=42,
        )
        test_records = defect_records.iloc[test_index]
        predictions = []
        probabilities = []
        for _, row in test_records.iterrows():
            result = predict_portable_cnn_defect_type(
                row["image_path"],
                spec.cnn_classifier_path,
                spec.cnn_classifier_path.with_suffix(".json"),
                defect_mask=row["mask"],
            )
            predictions.append(result["defect_type"])
            probabilities.append([result["class_probabilities"].get(label, 0.0) for label in label_order])
        return evaluated_subtype_metrics(
            test_records["label"].to_numpy(),
            np.asarray(predictions),
            np.asarray(probabilities, dtype=np.float64),
            label_order,
            protocol="Untouched stratified CNN test split evaluated with active OpenVINO anomaly masks",
            classifier_engine=engine,
            feature_mode="cnn_openvino_anomaly_crop",
        )

    bundle = joblib.load(spec.classifier_path)
    classifier = bundle["classifier"]
    feature_mode = str(bundle.get("feature_mode", "global"))
    if feature_mode == HANDCRAFTED_ROI_SHAPE_FEATURE_MODE:
        feature_extractor = preprocess = device = None
    else:
        feature_extractor, preprocess, device = shared_feature_runtime()
    features = _extract_training_features(
        defect_records,
        feature_mode,
        16,
        feature_extractor,
        preprocess,
        device,
    )
    if len(label_order) == 1:
        probabilities = np.ones((len(labels), 1), dtype=np.float64)
    else:
        folds = min(5, int(defect_records["label"].value_counts().min()))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        probabilities = cross_val_predict(
            clone(classifier),
            features,
            labels,
            cv=splitter,
            method="predict_proba",
            n_jobs=min(4, folds),
        )
    predictions = np.asarray(label_order)[np.argmax(probabilities, axis=1)]
    return evaluated_subtype_metrics(
        labels,
        predictions,
        probabilities,
        label_order,
        protocol=(
            "Stratified out-of-fold evaluation of the active classifier architecture using active OpenVINO anomaly masks"
        ),
        classifier_engine=engine,
        feature_mode=feature_mode,
    )


def metric_floor(metrics: dict) -> float:
    return min(float(metrics.get("accuracy", 0.0)), float(metrics.get("macro_f1", 0.0)))


def production_cv_metrics(metrics: dict) -> dict:
    """Return cross-validation metrics for the one classifier deployed in production."""
    evaluation = metrics.get("evaluation") or {}
    selected_name = evaluation.get("selected_classifier")
    selected_scores = (evaluation.get("candidate_scores") or {}).get(selected_name, {})
    if "mean_accuracy" not in selected_scores or "mean_macro_f1" not in selected_scores:
        return metrics
    return {
        "accuracy": float(selected_scores["mean_accuracy"]),
        "macro_f1": float(selected_scores["mean_macro_f1"]),
        "classifier": selected_name,
        "protocol": "Cross-validation estimate for the fixed production classifier",
    }


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
        "--evaluate-active-runtime",
        action="store_true",
        help="Evaluate the active subtype runtime with OpenVINO-generated masks and update metadata only.",
    )
    parser.add_argument(
        "--mask-source",
        choices=("ground_truth", "openvino"),
        default="ground_truth",
        help="Use annotation masks or production-style OpenVINO anomaly masks for ROI features.",
    )
    parser.add_argument(
        "--mask-policy",
        choices=("model", "p99.5", "p99", "p98", "p97", "p95"),
        default="model",
        help="Region used by subtype classification when OpenVINO masks are enabled.",
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
        if args.evaluate_active_runtime:
            if spec.openvino_path is None:
                raise FileNotFoundError(f"No OpenVINO model configured for {category}")
            print(f"Evaluating deployed subtype pipeline for {category}...", flush=True)
            deployed_records = attach_openvino_masks(
                records,
                spec.openvino_path,
                spec.input_size,
                category=category,
                mask_policy=args.mask_policy,
            )
            deployed_metrics = evaluate_active_runtime(category, deployed_records, spec)
            metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8-sig"))
            metadata["deployed_subtype_validation"] = deployed_metrics
            spec.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            calibration = deployed_metrics["confidence_calibration"]
            print(
                f"  accuracy={deployed_metrics['accuracy']}, macro_f1={deployed_metrics['macro_f1']}, "
                f"review_threshold={calibration['review_threshold']}, "
                f"calibration={calibration['method']}",
                flush=True,
            )
            continue
        if args.mask_source == "openvino":
            if spec.openvino_path is None:
                raise FileNotFoundError(f"No OpenVINO model configured for {category}")
            records = attach_openvino_masks(
                records,
                spec.openvino_path,
                spec.input_size,
                category=category,
                mask_policy=args.mask_policy,
            )
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
        current_metrics = (
            metadata.get("deployed_subtype_validation", {})
            if args.mask_source == "openvino"
            else metadata.get("defect_classifier", {})
        )
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
                    "classification_mask_policy": args.mask_policy if args.mask_source == "openvino" else "ground_truth",
                },
                defect_only=True,
                feature_mode=feature_mode,
                cross_validate_model=True,
            )
            metrics = result["metrics"]
            metrics["production_classifier_cv"] = production_cv_metrics(metrics)
            print(
                f"  {category}/{feature_mode}: accuracy={metrics['accuracy']}, "
                f"macro_f1={metrics['macro_f1']}, "
                f"production_macro_f1={metrics['production_classifier_cv']['macro_f1']}, "
                f"selected={metrics['evaluation']['selected_classifier']}",
                flush=True,
            )
            if best_result is None or is_better(
                metrics["production_classifier_cv"],
                best_result["metrics"]["production_classifier_cv"],
            ):
                best_result = result
                best_path = candidate_path

        assert best_result is not None and best_path is not None
        metadata["defect_classifier_revalidation"] = best_result["metrics"]
        should_promote = args.force or not current_metrics or is_better(
            best_result["metrics"]["production_classifier_cv"],
            current_metrics,
        )
        if should_promote:
            best_path.replace(spec.classifier_path)
            registry_path = registry_file()
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry_entry = registry.setdefault(category, {})
            registry_entry["cnn_classifier_path"] = None
            if spec.compact_classifier_path is not None and best_result["metrics"]["feature_mode"] == HANDCRAFTED_ROI_SHAPE_FEATURE_MODE:
                export_portable_forest(
                    best_result["bundle"]["classifier"],
                    spec.compact_classifier_path,
                    feature_mode=HANDCRAFTED_ROI_SHAPE_FEATURE_MODE,
                )
                registry.setdefault(category, {})["compact_classifier_path"] = str(
                    spec.compact_classifier_path.relative_to(PROJECT_ROOT)
                ).replace("\\", "/")
            else:
                registry_entry["compact_classifier_path"] = None
            registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
            metadata["defect_classifier"] = best_result["metrics"]
            status = "promoted"
        else:
            status = "kept-existing"

        spec.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

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
