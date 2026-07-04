from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models

LABEL_ORDER = ["good", "broken_large", "broken_small", "contamination"]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet18_feature_extractor(device: torch.device | None = None):
    """Build a frozen ImageNet ResNet18 feature extractor."""
    device = device or get_device()
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval().to(device)
    preprocess = weights.transforms()
    return feature_extractor, preprocess, device


def load_rgb_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def extract_features(
    image_paths: list[str | Path],
    batch_size: int = 16,
    feature_extractor=None,
    preprocess=None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Extract ResNet18 embeddings for a list of image paths."""
    if feature_extractor is None or preprocess is None:
        feature_extractor, preprocess, device = build_resnet18_feature_extractor(device)
    else:
        device = device or get_device()

    features = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = [preprocess(load_rgb_image(path)) for path in batch_paths]
            batch = torch.stack(tensors).to(device)
            embeddings = feature_extractor(batch)
            embeddings = embeddings.flatten(start_dim=1).cpu().numpy()
            features.append(embeddings)

    return np.vstack(features)


def create_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_defect_classifier(
    dataset_df: pd.DataFrame,
    output_path: str | Path,
    test_size: float = 0.3,
    random_state: int = 42,
    batch_size: int = 16,
) -> dict:
    """Train and save a defect type classifier using frozen ResNet18 features."""
    data = dataset_df[dataset_df["label"].isin(LABEL_ORDER)].copy()
    if data.empty:
        raise ValueError("Dataset is empty. Cannot train classifier.")

    train_df, eval_df = train_test_split(
        data,
        test_size=test_size,
        stratify=data["label"],
        random_state=random_state,
    )

    feature_extractor, preprocess, device = build_resnet18_feature_extractor()

    x_train = extract_features(
        train_df["image_path"].tolist(),
        batch_size=batch_size,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )
    x_eval = extract_features(
        eval_df["image_path"].tolist(),
        batch_size=batch_size,
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )

    y_train = train_df["label"].tolist()
    y_eval = eval_df["label"].tolist()

    classifier = create_classifier()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_eval)
    labels = [label for label in LABEL_ORDER if label in sorted(data["label"].unique())]
    metrics = {
        "accuracy": round(float(accuracy_score(y_eval, y_pred)), 4),
        "classification_report": classification_report(
            y_eval,
            y_pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_eval, y_pred, labels=labels).tolist(),
        "labels": labels,
        "train_size": int(len(train_df)),
        "eval_size": int(len(eval_df)),
        "feature_extractor": "resnet18_imagenet1k_v1",
    }

    bundle = {
        "classifier": classifier,
        "labels": labels,
        "metrics": metrics,
        "feature_extractor": "resnet18_imagenet1k_v1",
        "image_size": [224, 224],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)

    return {
        "bundle": bundle,
        "train_df": train_df,
        "eval_df": eval_df,
        "y_eval": y_eval,
        "y_pred": y_pred,
        "metrics": metrics,
        "output_path": output_path,
    }


def load_classifier_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def predict_defect_type(image_path: str | Path, classifier_bundle: dict) -> dict:
    feature_extractor, preprocess, device = build_resnet18_feature_extractor()
    features = extract_features(
        [image_path],
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )
    classifier = classifier_bundle["classifier"]
    label = str(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]
    confidence = float(np.max(probabilities))
    class_probabilities = {
        str(class_name): float(probability)
        for class_name, probability in zip(classifier.classes_, probabilities, strict=False)
    }

    return {
        "defect_type": label,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probabilities,
    }
