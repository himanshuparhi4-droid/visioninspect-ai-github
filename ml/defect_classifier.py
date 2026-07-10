from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


class DefectClassifierError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def load_classifier_runtime(path_value: str) -> tuple[dict, object, object, object]:
    from ml.classifier import build_resnet18_feature_extractor, load_classifier_bundle

    path = Path(path_value)
    if not path.exists():
        raise DefectClassifierError(f"Classifier artifact not found: {path}")

    bundle = load_classifier_bundle(path)
    feature_extractor, preprocess, device = build_resnet18_feature_extractor()
    return bundle, feature_extractor, preprocess, device


def classify_defect_type(image_path: str | Path, classifier_model_path: str | Path) -> dict:
    from ml.classifier import extract_features

    bundle, feature_extractor, preprocess, device = load_classifier_runtime(str(classifier_model_path))
    features = extract_features(
        [image_path],
        feature_extractor=feature_extractor,
        preprocess=preprocess,
        device=device,
    )
    classifier = bundle["classifier"]
    label = str(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]
    class_probabilities = {
        str(class_name): round(float(probability), 4)
        for class_name, probability in zip(classifier.classes_, probabilities, strict=False)
    }

    return {
        "defect_type": label,
        "confidence": round(float(np.max(probabilities)), 4),
        "class_probabilities": class_probabilities,
    }
