"""Calibrate a category anomaly-model threshold using a stratified development split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.classifier import export_portable_forest
from ml.config import MODELS_DIR, MVTEC_DATASET_ROOT
from ml.model_registry import SUPPORTED_CATEGORIES, category_model_spec
from ml.padim_detector import load_openvino_calibrator, load_openvino_runtime, openvino_spatial_features


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


def openvino_calibrator_candidates(*, include_linear: bool = False) -> dict[str, object]:
    common = {
        "n_estimators": 300,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    candidates = {
        "extra_trees_sqrt_leaf1": ExtraTreesClassifier(
            min_samples_leaf=1,
            max_features="sqrt",
            **common,
        ),
        "extra_trees_wide_leaf1": ExtraTreesClassifier(
            min_samples_leaf=1,
            max_features=0.75,
            **common,
        ),
        "extra_trees_sqrt_leaf2": ExtraTreesClassifier(
            min_samples_leaf=2,
            max_features="sqrt",
            **common,
        ),
        "random_forest_sqrt_leaf1": RandomForestClassifier(
            min_samples_leaf=1,
            max_features="sqrt",
            **common,
        ),
    }
    if include_linear:
        candidates = {
            "logistic_c0.1": make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42),
            ),
            "logistic_c1": make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
            ),
            "logistic_c10": make_pipeline(
                StandardScaler(),
                LogisticRegression(C=10.0, class_weight="balanced", max_iter=2000, random_state=42),
            ),
            **candidates,
        }
    return candidates


def best_joint_threshold(probabilities: np.ndarray, targets: np.ndarray) -> float:
    _, _, roc_thresholds = roc_curve(targets, probabilities)
    candidates = np.unique(
        np.concatenate(
            [
                roc_thresholds[np.isfinite(roc_thresholds)],
                np.linspace(0.20, 0.80, 25),
                np.asarray([0.5]),
            ]
        )
    )
    best_threshold = 0.5
    best_score = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        balanced = float(balanced_accuracy_score(targets, predictions))
        f1 = float(f1_score(targets, predictions, zero_division=0))
        score = (min(balanced, f1), (balanced + f1) / 2.0, -abs(float(threshold) - 0.5))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def calibrator_candidate_scores(
    candidates: dict[str, object],
    features: np.ndarray,
    targets: np.ndarray,
    splitter: StratifiedKFold,
) -> dict[str, dict]:
    scores: dict[str, dict] = {}
    for name, candidate in candidates.items():
        probabilities = cross_val_predict(
            clone(candidate),
            features,
            targets,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = best_joint_threshold(probabilities, targets)
        predictions = (probabilities >= threshold).astype(int)
        scores[name] = {
            "threshold": round(float(threshold), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(targets, predictions)), 4),
            "f1": round(float(f1_score(targets, predictions, zero_division=0)), 4),
            "auroc": round(float(roc_auc_score(targets, probabilities)), 4),
        }
    return scores


def best_calibrator_name(scores: dict[str, dict]) -> str:
    return max(
        scores,
        key=lambda name: (
            min(scores[name]["balanced_accuracy"], scores[name]["f1"]),
            scores[name]["auroc"],
            scores[name]["balanced_accuracy"] + scores[name]["f1"],
        ),
    )


def calibrate_openvino_category(
    category: str,
    dataset_root: Path,
    *,
    include_linear: bool = False,
    fixed_calibrator: str | None = None,
    fixed_threshold: float | None = None,
    artifact_path: Path | None = None,
) -> dict:
    rows = openvino_feature_rows(category, dataset_root)
    features = np.vstack([row["features"] for row in rows])
    targets = np.asarray([row["target"] for row in rows], dtype=int)
    class_counts = np.bincount(targets)
    folds = min(5, int(class_counts.min()))
    if folds < 2:
        raise ValueError(f"{category} does not contain enough good and defective images for calibration")

    candidates = openvino_calibrator_candidates(include_linear=include_linear)
    if fixed_calibrator:
        if fixed_calibrator not in candidates:
            raise ValueError(f"Unknown OpenVINO calibrator: {fixed_calibrator}")
        candidates = {fixed_calibrator: candidates[fixed_calibrator]}
    outer_splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    probabilities = np.zeros(len(targets), dtype=np.float32)
    predictions = np.zeros(len(targets), dtype=int)
    outer_selections: list[str] = []
    outer_thresholds: list[float] = []
    for outer_fold, (train_index, test_index) in enumerate(outer_splitter.split(features, targets), start=1):
        if fixed_threshold is not None:
            selected_name = next(iter(candidates))
            selected_threshold = fixed_threshold
        else:
            inner_counts = np.bincount(targets[train_index])
            inner_folds = min(4, int(inner_counts.min()))
            inner_splitter = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=42 + outer_fold,
            )
            inner_scores = calibrator_candidate_scores(
                candidates,
                features[train_index],
                targets[train_index],
                inner_splitter,
            )
            selected_name = best_calibrator_name(inner_scores)
            selected_threshold = float(inner_scores[selected_name]["threshold"])
        fold_classifier = clone(candidates[selected_name])
        fold_classifier.fit(features[train_index], targets[train_index])
        fold_probabilities = fold_classifier.predict_proba(features[test_index])[:, 1]
        probabilities[test_index] = fold_probabilities
        predictions[test_index] = (fold_probabilities >= selected_threshold).astype(int)
        outer_selections.append(selected_name)
        outer_thresholds.append(selected_threshold)

    # Pick and fit the runtime calibrator only after nested evaluation is complete.
    production_splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=143)
    candidate_scores = calibrator_candidate_scores(candidates, features, targets, production_splitter)
    production_name = best_calibrator_name(candidate_scores)
    decision_threshold = (
        float(fixed_threshold)
        if fixed_threshold is not None
        else float(candidate_scores[production_name]["threshold"])
    )
    classifier = clone(candidates[production_name])
    classifier.fit(features, targets)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        targets, predictions, labels=[0, 1]
    ).ravel()
    spec = category_model_spec(category)
    if spec.openvino_calibrator_path is None:
        raise RuntimeError(f"No OpenVINO calibrator path configured for {category}")
    output_path = artifact_path or spec.openvino_calibrator_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_portable_forest(
        classifier,
        output_path,
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
            "nested stratified cross-validation on labelled MVTec test images; each outer fold performs "
            + (
                "calibrator and threshold selection using only its training partition; "
                if fixed_threshold is None
                else f"evaluation with a pre-specified {fixed_threshold:.4f} decision threshold; "
            )
            + "production calibrator is fitted on all labelled images after evaluation"
        ),
        "folds": folds,
        "samples": int(len(targets)),
        "selected_classifier": production_name,
        "candidate_scores": candidate_scores,
        "outer_selection_frequency": {
            name: outer_selections.count(name) for name in sorted(set(outer_selections))
        },
        "outer_thresholds": [round(float(value), 6) for value in outer_thresholds],
        "decision_threshold": round(float(decision_threshold), 6),
        "accuracy": round(float(accuracy_score(targets, predictions)), 4),
        "precision": round(float(precision_score(targets, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(targets, predictions, zero_division=0)), 4),
        "specificity": round(float(true_negative / max(true_negative + false_positive, 1)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(targets, predictions)), 4),
        "f1": round(float(f1_score(targets, predictions, zero_division=0)), 4),
        "auroc": round(float(roc_auc_score(targets, probabilities)), 4),
        "confusion_matrix": [
            [int(true_negative), int(false_positive)],
            [int(false_negative), int(true_positive)],
        ],
        "defect_subtype_recall": subtype_recall,
        "artifact": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def calibration_metric_floor(result: dict) -> float:
    return min(float(result.get("balanced_accuracy", 0.0)), float(result.get("f1", 0.0)))


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
        "--include-linear-calibrators",
        action="store_true",
        help="Also evaluate portable regularized logistic calibrators for OpenVINO outputs.",
    )
    parser.add_argument(
        "--fixed-calibrator",
        help="Evaluate one named OpenVINO calibrator while tuning only its decision threshold.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        help="Use one pre-specified OpenVINO decision threshold in every held-out fold.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing calibration even if it is weaker.")
    parser.add_argument(
        "--objective",
        choices=("f1", "balanced_accuracy"),
        default="f1",
        help="Metric optimized on the calibration split when choosing the decision threshold.",
    )
    args = parser.parse_args()
    if args.fixed_threshold is not None and not args.fixed_calibrator:
        parser.error("--fixed-threshold requires --fixed-calibrator")
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
        try:
            if args.engine == "baseline":
                result = calibrate_baseline_category(category, dataset_root, args.objective)
            elif args.engine == "openvino":
                spec = category_model_spec(category)
                if spec.openvino_calibrator_path is None:
                    raise RuntimeError(f"No OpenVINO calibrator path configured for {category}")
                candidate_path = spec.openvino_calibrator_path.with_name(
                    f"{spec.openvino_calibrator_path.stem}.candidate{spec.openvino_calibrator_path.suffix}"
                )
                result = calibrate_openvino_category(
                    category,
                    dataset_root,
                    include_linear=args.include_linear_calibrators,
                    fixed_calibrator=args.fixed_calibrator,
                    fixed_threshold=args.fixed_threshold,
                    artifact_path=candidate_path,
                )
            else:
                result = calibrate_category(category, dataset_root, args.objective)
        finally:
            # Category OpenVINO graphs can approach a gigabyte each. Release the
            # previous graph before loading the next category in long calibration runs.
            load_openvino_runtime.cache_clear()
            load_openvino_calibrator.cache_clear()
        spec = category_model_spec(category)
        metadata = json.loads(spec.metadata_path.read_text(encoding="utf-8"))
        metadata_key = {
            "baseline": "baseline_threshold_calibration",
            "advanced": "threshold_calibration",
            "openvino": "openvino_spatial_calibration",
        }[args.engine]
        if args.engine == "openvino":
            current = metadata.get(metadata_key, {})
            should_promote = args.force or not current or calibration_metric_floor(result) > calibration_metric_floor(current)
            if should_promote:
                candidate_path.replace(spec.openvino_calibrator_path)
                result["artifact"] = str(spec.openvino_calibrator_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            else:
                candidate_path.unlink(missing_ok=True)
                print(
                    f"Kept existing {category} calibration: candidate floor={calibration_metric_floor(result):.4f}, "
                    f"current floor={calibration_metric_floor(current):.4f}",
                    flush=True,
                )
                result = current
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
