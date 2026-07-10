from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from ml.classifier import LABEL_ORDER, build_resnet18_feature_extractor, create_classifier, extract_features
from ml.config import RAW_DATA_DIR
from ml.dataset_loader import load_bottle_dataframe


def prepare_classifier_dataframe(root: Path, max_images_per_class: int | None = None):
    df = load_bottle_dataframe(root)
    if "label" not in df.columns:
        raise ValueError(f"No classifier images found under {root}")

    data = df[df["label"].isin(LABEL_ORDER)].copy()
    data = data[data["image_path"].notna()]

    if max_images_per_class:
        sampled_groups = [
            group.sample(min(len(group), max_images_per_class), random_state=42) for _, group in data.groupby("label")
        ]
        data = pd.concat(sampled_groups, ignore_index=True) if sampled_groups else data.iloc[0:0].copy()

    if data.empty:
        raise ValueError(f"No classifier images found under {root}")
    return data.reset_index(drop=True)


def run_stratified_kfold(
    root: Path = RAW_DATA_DIR,
    folds: int = 5,
    batch_size: int = 16,
    max_images_per_class: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    data = prepare_classifier_dataframe(root, max_images_per_class=max_images_per_class)
    labels = [label for label in LABEL_ORDER if label in sorted(data["label"].unique())]
    counts = data["label"].value_counts().to_dict()
    min_class_count = min(counts.values())
    effective_folds = min(folds, min_class_count)

    if effective_folds < 2:
        raise ValueError(f"K-fold validation needs at least 2 images per class. Counts: {counts}")

    feature_extractor, preprocess, device = build_resnet18_feature_extractor()
    features = extract_features(
        data["image_path"].tolist(),
        batch_size=batch_size,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )

    y = data["label"].to_numpy()
    splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)

    fold_results = []
    all_true: list[str] = []
    all_pred: list[str] = []

    for fold_index, (train_index, eval_index) in enumerate(splitter.split(features, y), start=1):
        classifier = create_classifier()
        classifier.fit(features[train_index], y[train_index])
        predictions = classifier.predict(features[eval_index])
        truth = y[eval_index]

        all_true.extend(truth.tolist())
        all_pred.extend(predictions.tolist())

        fold_results.append(
            {
                "fold": fold_index,
                "train_size": int(len(train_index)),
                "eval_size": int(len(eval_index)),
                "accuracy": round(float(accuracy_score(truth, predictions)), 4),
                "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(),
                "classification_report": classification_report(
                    truth,
                    predictions,
                    labels=labels,
                    zero_division=0,
                    output_dict=True,
                ),
            }
        )

    accuracies = [fold["accuracy"] for fold in fold_results]
    summary = {
        "validation_type": "stratified_kfold_classifier_validation",
        "dataset": str(root),
        "labels": labels,
        "class_counts": counts,
        "requested_folds": folds,
        "effective_folds": effective_folds,
        "total_images": int(len(data)),
        "feature_extractor": "resnet18_imagenet1k_v1",
        "classifier": "standard_scaler_logistic_regression_balanced",
        "folds": fold_results,
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "std_accuracy": round(float(np.std(accuracies)), 4),
        "aggregate_confusion_matrix": confusion_matrix(all_true, all_pred, labels=labels).tolist(),
        "aggregate_classification_report": classification_report(
            all_true,
            all_pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "kfold_classifier_validation.json"
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["output_path"] = str(output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stratified K-fold validation for the defect classifier.")
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for JSON export.")
    args = parser.parse_args()

    summary = run_stratified_kfold(
        root=args.data_root,
        folds=args.folds,
        batch_size=args.batch_size,
        max_images_per_class=args.max_images_per_class,
        output_dir=args.output_dir,
    )

    print(
        json.dumps(
            {
                "effective_folds": summary["effective_folds"],
                "total_images": summary["total_images"],
                "class_counts": summary["class_counts"],
                "mean_accuracy": summary["mean_accuracy"],
                "std_accuracy": summary["std_accuracy"],
                "output_path": summary.get("output_path"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
