"""Calibrate a category anomaly-model threshold using a stratified development split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.classifier import export_portable_forest
from ml.config import MODELS_DIR, MVTEC_DATASET_ROOT
from ml.model_registry import SUPPORTED_CATEGORIES, category_model_spec
from ml.padim_detector import load_openvino_runtime, openvino_spatial_features


def baseline_score_rows(category: str, dataset_root: Path) -> list[dict]:
    import cv2

    from ml.baseline_detector import (
        anomaly_score,
        embedding_anomaly_scores,
        load_reference_profile,
        normalized_anomaly_map,
    )
    from ml.classifier import extract_features
    from ml.defect_classifier import shared_feature_runtime

    spec = category_model_spec(category)
    profile = load_reference_profile(spec.baseline_profile_path)
    if "embedding_bank" not in profile:
        raise RuntimeError(f"{category} has no embedding bank. Run train_category_models.py --portable-only first.")

    paths = []
    targets = []
    for label_dir in sorted(path for path in (dataset_root / category / "test").iterdir() if path.is_dir()):
        image_paths = sorted(label_dir.glob("*.png"))
        paths.extend(image_paths)
        targets.extend([int(label_dir.name != "good")] * len(image_paths))

    feature_extractor, preprocess, device = shared_feature_runtime()
    features = extract_features(
        paths,
        batch_size=32,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )
    embedding_scores = embedding_anomaly_scores(features, profile["embedding_bank"])

    rows = []
    for path, target, embedding_score in zip(paths, targets, embedding_scores, strict=True):
        image = cv2.imread(str(path))
        residual_map = normalized_anomaly_map(image, profile)
        residual_score = anomaly_score(residual_map, mask=profile["foreground_mask"])
        rows.append(
            {
                "path": str(path),
                "score": float(embedding_score),
                "residual_score": float(residual_score),
                "target": target,
            }
        )
    return rows


def category_score_rows(category: str, dataset_root: Path) -> list[dict]:
    import torch
    from anomalib.engine import Engine

    from ml.padim_detector import load_anomaly_runtime

    spec = category_model_spec(category)
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {category}: {spec.checkpoint_path}")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    model, _ = load_anomaly_runtime(
        str(spec.checkpoint_path), spec.model_kind, "gpu" if torch.cuda.is_available() else "cpu"
    )
    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=False, enable_progress_bar=False
    )
    predictions = engine.predict(model=model, data_path=dataset_root / category / "test", return_predictions=True)
    rows = []
    for item in predictions:
        path = Path(item.image_path[0])
        score = float(item.pred_score.detach().cpu().numpy().reshape(-1)[0])
        rows.append({"path": str(path), "score": score, "target": int(path.parent.name != "good")})
    return rows


def openvino_feature_rows(category: str, dataset_root: Path) -> list[dict]:
    import cv2

    spec = category_model_spec(category)
    if spec.openvino_path is None or not spec.openvino_path.exists():
        raise FileNotFoundError(f"OpenVINO model not found for {category}: {spec.openvino_path}")
    compiled_model = load_openvino_runtime(str(spec.openvino_path), "CPU")
    rows = []
    test_root = dataset_root / category / "test"
    for label_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        paths = sorted(label_dir.glob("*.png"))
        for start in range(0, len(paths), 8):
            batch_paths = paths[start : start + 8]
            batch = []
            for path in batch_paths:
                image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
                batch.append(image.transpose(2, 0, 1))
            outputs = compiled_model([np.stack(batch)])
            scores = np.asarray(outputs[compiled_model.output("pred_score")]).reshape(-1)
            anomaly_maps = np.asarray(outputs[compiled_model.output("anomaly_map")])[:, 0]
            for path, score, anomaly_map in zip(batch_paths, scores, anomaly_maps, strict=True):
                rows.append(
                    {
                        "path": str(path),
                        "label": label_dir.name,
                        "target": int(label_dir.name != "good"),
                        "features": openvino_spatial_features(float(score), anomaly_map),
                    }
                )
    return rows


def best_f1_threshold(scores: np.ndarray, targets: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def best_balanced_threshold(scores: np.ndarray, targets: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(targets, scores)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return best_f1_threshold(scores, targets)
    objective = true_positive_rate[finite] - false_positive_rate[finite]
    return float(thresholds[finite][int(np.nanargmax(objective))])


def calibration_split(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(targets))
    return train_test_split(indices, test_size=0.7, stratify=targets, random_state=42)


def calibrated_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    calibration_index: np.ndarray,
    holdout_index: np.ndarray,
    objective: str,
) -> dict:
    if objective == "f1":
        threshold = best_f1_threshold(scores[calibration_index], targets[calibration_index])
    else:
        threshold = best_balanced_threshold(scores[calibration_index], targets[calibration_index])
    holdout_predictions = (scores[holdout_index] > threshold).astype(int)
    holdout_targets = targets[holdout_index]
    true_negative, false_positive, _, _ = confusion_matrix(
        holdout_targets,
        holdout_predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "threshold": round(threshold, 6),
        "objective": objective,
        "protocol": (
            "30% stratified development calibration and 70% untouched holdout evaluation "
            "from labelled MVTec test folders; threshold selected on calibration split only"
        ),
        "calibration_samples": int(len(calibration_index)),
        "holdout_samples": int(len(holdout_index)),
        "holdout_accuracy": round(float(accuracy_score(holdout_targets, holdout_predictions)), 4),
        "holdout_precision": round(float(precision_score(holdout_targets, holdout_predictions, zero_division=0)), 4),
        "holdout_recall": round(float(recall_score(holdout_targets, holdout_predictions, zero_division=0)), 4),
        "holdout_specificity": round(
            float(true_negative / max(true_negative + false_positive, 1)),
            4,
        ),
        "holdout_balanced_accuracy": round(
            float(balanced_accuracy_score(holdout_targets, holdout_predictions)),
            4,
        ),
        "holdout_f1": round(float(f1_score(holdout_targets, holdout_predictions, zero_division=0)), 4),
        "holdout_auroc": round(float(roc_auc_score(holdout_targets, scores[holdout_index])), 4),
    }


def calibrate_baseline_category(category: str, dataset_root: Path, objective: str) -> dict:
    rows = baseline_score_rows(category, dataset_root)
    targets = np.asarray([row["target"] for row in rows], dtype=int)
    embedding_scores = np.asarray([row["score"] for row in rows], dtype=np.float32)
    residual_scores = np.asarray([row["residual_score"] for row in rows], dtype=np.float32)
    calibration_index, holdout_index = calibration_split(targets)
    result = calibrated_threshold(embedding_scores, targets, calibration_index, holdout_index, objective)
    result.update(
        {
            "detector": "resnet18_normal_memory",
            "localization": "opencv_normalized_residual",
            "residual_threshold": round(
                best_f1_threshold(residual_scores[calibration_index], targets[calibration_index]),
                6,
            ),
        }
    )
    return result


def calibrate_category(category: str, dataset_root: Path, objective: str) -> dict:
    rows = category_score_rows(category, dataset_root)
    targets = np.asarray([row["target"] for row in rows])
    calibration_index, holdout_index = calibration_split(targets)
    scores = np.asarray([row["score"] for row in rows])
    result = calibrated_threshold(scores, targets, calibration_index, holdout_index, objective)
    return {
        "model_kind": category_model_spec(category).model_kind,
        **result,
    }


def calibrate_openvino_category(category: str, dataset_root: Path) -> dict:
    rows = openvino_feature_rows(category, dataset_root)
    features = np.vstack([row["features"] for row in rows])
    targets = np.asarray([row["target"] for row in rows], dtype=int)
    classifier = ExtraTreesClassifier(
        n_estimators=800,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = cross_val_predict(classifier, features, targets, cv=splitter, n_jobs=1)
    decision_threshold = 0.5
    classifier.fit(features, targets)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        targets, predictions, labels=[0, 1]
    ).ravel()
    spec = category_model_spec(category)
    if spec.openvino_calibrator_path is None:
        raise RuntimeError(f"No OpenVINO calibrator path configured for {category}")
    spec.openvino_calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    export_portable_forest(
        classifier,
        spec.openvino_calibrator_path,
        feature_mode="openvino_spatial_v1",
        decision_threshold=decision_threshold,
    )
    subtype_recall = {}
    labels = np.asarray([row["label"] for row in rows])
    for label in sorted(set(labels) - {"good"}):
        subtype_recall[label] = round(float(predictions[labels == label].mean()), 4)
    return {
        "detector": f"{spec.model_kind}_openvino_spatial_calibrator",
        "protocol": (
            "5-fold stratified out-of-fold evaluation on labelled MVTec test images; "
            "production calibrator fitted on all labelled images"
        ),
        "folds": 5,
        "samples": int(len(targets)),
        "decision_threshold": round(float(decision_threshold), 6),
        "accuracy": round(float(accuracy_score(targets, predictions)), 4),
        "precision": round(float(precision_score(targets, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(targets, predictions, zero_division=0)), 4),
        "specificity": round(float(true_negative / max(true_negative + false_positive, 1)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(targets, predictions)), 4),
        "f1": round(float(f1_score(targets, predictions, zero_division=0)), 4),
        "confusion_matrix": [
            [int(true_negative), int(false_positive)],
            [int(false_negative), int(true_positive)],
        ],
        "defect_subtype_recall": subtype_recall,
        "artifact": str(spec.openvino_calibrator_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default="all")
    parser.add_argument("--dataset-root", type=Path, default=MVTEC_DATASET_ROOT)
    parser.add_argument(
        "--engine",
        choices=("baseline", "advanced", "openvino"),
        default="baseline",
        help="Calibrate the portable baseline or the optional advanced anomaly model.",
    )
    parser.add_argument(
        "--objective",
        choices=("f1", "balanced_accuracy"),
        default="f1",
        help="Metric optimized on the calibration split when choosing the decision threshold.",
    )
    args = parser.parse_args()
    categories = (
        SUPPORTED_CATEGORIES
        if args.categories == "all"
        else tuple(item.strip().lower().replace("-", "_") for item in args.categories.split(",") if item.strip())
    )
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    registry_path = MODELS_DIR / "category_model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}

    for category in categories:
        print(f"Calibrating {args.engine} detector for {category}...", flush=True)
        if args.engine == "baseline":
            result = calibrate_baseline_category(category, dataset_root, args.objective)
        elif args.engine == "openvino":
            result = calibrate_openvino_category(category, dataset_root)
        else:
            result = calibrate_category(category, dataset_root, args.objective)
        spec = category_model_spec(category)
        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
        metadata_key = {
            "baseline": "baseline_threshold_calibration",
            "advanced": "threshold_calibration",
            "openvino": "openvino_spatial_calibration",
        }[args.engine]
        metadata[metadata_key] = result
        spec.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        entry = registry.setdefault(category, {})
        if args.engine == "baseline":
            entry["baseline_score_threshold"] = result["threshold"]
            entry["baseline_residual_threshold"] = result["residual_threshold"]
        elif args.engine == "advanced":
            entry["padim_score_threshold"] = result["threshold"]
        else:
            entry["openvino_calibrator_path"] = result["artifact"]
        print(json.dumps({category: result}, indent=2), flush=True)

    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
