"""Promote optimal classifiers cleanly into model artifacts, metadata, and registry."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ml.classifier import (
    ROI_SHAPE_TEXTURE_EXTRACTOR_NAME,
    ROI_SHAPE_TEXTURE_FEATURE_MODE,
    ROI_TEXTURE_EXTRACTOR_NAME,
    ROI_TEXTURE_FEATURE_MODE,
    _extract_training_features,
    build_resnet18_feature_extractor,
)
from ml.config import MVTEC_DATASET_ROOT


def load_category_data(category: str):
    root = MVTEC_DATASET_ROOT / category
    records = []
    test_dir = root / "test"
    for label_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        if label_dir.name == "good":
            continue
        for img_path in sorted(label_dir.glob("*.png")):
            mask_path = root / "ground_truth" / label_dir.name / f"{img_path.stem}_mask.png"
            records.append({
                "label": label_dir.name,
                "image_path": str(img_path),
                "mask_path": str(mask_path) if mask_path.exists() else None,
            })
    return pd.DataFrame(records)


def promote_wood(ext, prep, dev):
    print("\n--- Promoting Wood ---")
    df = load_category_data("wood")
    labels = sorted(df["label"].unique().tolist())
    y = df["label"].to_numpy()
    feature_mode = ROI_TEXTURE_FEATURE_MODE
    feats = _extract_training_features(df, feature_mode, 16, ext, prep, dev)

    base = LinearSVC(C=0.01, random_state=42, max_iter=4000)
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(pipeline, feats, y, cv=skf)

    acc = round(float(accuracy_score(y, preds)), 4)
    f1 = round(float(f1_score(y, preds, average="macro")), 4)
    report = classification_report(y, preds, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y, preds, labels=labels).tolist()

    pipeline.fit(feats, y)

    metrics = {
        "accuracy": acc,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "train_size": len(df),
        "eval_size": len(df),
        "feature_extractor": ROI_TEXTURE_EXTRACTOR_NAME,
        "feature_mode": feature_mode,
        "defect_only": True,
        "evaluation": {
            "protocol": "stratified 5-fold cross-validation; calibrated linear maximum-margin classifier fitted on all labelled defect images",
            "selected_classifier": "calibrated_linear_svc_c0.01",
            "folds": 5,
        },
        "dataset_context": {
            "source": "MVTec AD labelled test folders",
            "protocol": "defect-only subtype classification with stratified cross-validation using ground truth masks",
            "category": "wood",
        },
    }

    bundle = {
        "classifier": pipeline,
        "labels": labels,
        "metrics": metrics,
        "feature_extractor": ROI_TEXTURE_EXTRACTOR_NAME,
        "feature_mode": feature_mode,
        "defect_only": True,
        "image_size": [224, 224],
        "dataset_context": metrics["dataset_context"],
    }

    out_path = PROJECT_ROOT / "models" / "categories" / "wood" / "defect_classifier.pkl"
    joblib.dump(bundle, out_path)

    meta_path = PROJECT_ROOT / "models" / "categories" / "wood" / "model_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["defect_classifier"] = metrics
    meta.pop("defect_classifier_revalidation", None)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wood promoted: accuracy={acc}, macro_f1={f1}, model=calibrated_linear_svc_c0.01")


def promote_capsule(ext, prep, dev):
    print("\n--- Promoting Capsule ---")
    df = load_category_data("capsule")
    labels = sorted(df["label"].unique().tolist())
    y = df["label"].to_numpy()
    feature_mode = ROI_SHAPE_TEXTURE_FEATURE_MODE
    feats = _extract_training_features(df, feature_mode, 16, ext, prep, dev)

    base = LinearSVC(C=0.01, random_state=42, max_iter=4000)
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(pipeline, feats, y, cv=skf)

    acc = round(float(accuracy_score(y, preds)), 4)
    f1 = round(float(f1_score(y, preds, average="macro")), 4)
    report = classification_report(y, preds, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y, preds, labels=labels).tolist()

    pipeline.fit(feats, y)

    metrics = {
        "accuracy": acc,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "train_size": len(df),
        "eval_size": len(df),
        "feature_extractor": ROI_SHAPE_TEXTURE_EXTRACTOR_NAME,
        "feature_mode": feature_mode,
        "defect_only": True,
        "evaluation": {
            "protocol": "stratified 5-fold cross-validation; calibrated linear maximum-margin classifier fitted on all labelled defect images",
            "selected_classifier": "calibrated_linear_svc_c0.01",
            "folds": 5,
        },
        "dataset_context": {
            "source": "MVTec AD labelled test folders",
            "protocol": "defect-only subtype classification with stratified cross-validation using ground truth masks",
            "category": "capsule",
        },
    }

    bundle = {
        "classifier": pipeline,
        "labels": labels,
        "metrics": metrics,
        "feature_extractor": ROI_SHAPE_TEXTURE_EXTRACTOR_NAME,
        "feature_mode": feature_mode,
        "defect_only": True,
        "image_size": [224, 224],
        "dataset_context": metrics["dataset_context"],
    }

    out_path = PROJECT_ROOT / "models" / "categories" / "capsule" / "defect_classifier.pkl"
    joblib.dump(bundle, out_path)

    meta_path = PROJECT_ROOT / "models" / "categories" / "capsule" / "model_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["defect_classifier"] = metrics
    meta.pop("defect_classifier_revalidation", None)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Capsule promoted: accuracy={acc}, macro_f1={f1}, model=calibrated_linear_svc_c0.01")


def update_registry():
    print("\n--- Updating Registry ---")
    reg_path = PROJECT_ROOT / "models" / "category_model_registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))

    # Ensure carpet and bottle have CNN active
    reg["carpet"]["cnn_classifier_path"] = "models/categories/carpet/cnn_defect_classifier.onnx"
    reg["bottle"]["cnn_classifier_path"] = "models/categories/bottle/cnn_defect_classifier.onnx"

    # Wood, capsule, grid, zipper, pill use high-accuracy standard bundles
    for c in ["wood", "capsule", "grid", "zipper", "pill"]:
        if c in reg:
            reg[c].pop("cnn_classifier_path", None)
            reg[c].pop("compact_classifier_path", None)

    reg_path.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
    print("Registry successfully updated.")


def main():
    ext, prep, dev = build_resnet18_feature_extractor()
    promote_wood(ext, prep, dev)
    promote_capsule(ext, prep, dev)
    update_registry()
    print("\nAll optimal model promotions complete!")


if __name__ == "__main__":
    main()
