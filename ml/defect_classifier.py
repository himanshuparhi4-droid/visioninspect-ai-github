from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ml.classifier import (
    GLOBAL_TEXTURE_FEATURE_MODE,
    ROI_PIXEL_TEXTURE_FEATURE_MODE,
    ROI_SHAPE_TEXTURE_FEATURE_MODE,
    ROI_TEXTURE_FEATURE_MODE,
)


class DefectClassifierError(RuntimeError):
    pass


@lru_cache(maxsize=32)
def load_classifier_runtime(path_value: str) -> dict:
    from ml.classifier import load_classifier_bundle

    path = Path(path_value)
    if not path.exists():
        raise DefectClassifierError(f"Classifier artifact not found: {path}")
    return load_classifier_bundle(path)


@lru_cache(maxsize=1)
def shared_feature_runtime():
    from ml.classifier import build_resnet18_feature_extractor

    return build_resnet18_feature_extractor()


def classify_defect_type(
    image_path: str | Path,
    classifier_model_path: str | Path,
    defect_mask: np.ndarray | None = None,
) -> dict:
    from ml.classifier import (
        extract_features,
        extract_global_texture_features,
        extract_roi_pixel_texture_features,
        extract_roi_shape_texture_features,
        extract_roi_texture_features,
    )

    classifier_model_path = Path(classifier_model_path)
    cnn_path = classifier_model_path.with_name("cnn_defect_classifier.pt")
    if cnn_path.exists():
        try:
            from ml.cnn_classifier import predict_cnn_defect_type

            return predict_cnn_defect_type(image_path, cnn_path, defect_mask=defect_mask)
        except Exception:
            # Keep the production workflow available if an experimental CNN
            # artifact is corrupted or incompatible with the local runtime.
            pass

    bundle = load_classifier_runtime(str(classifier_model_path))
    feature_extractor, preprocess, device = shared_feature_runtime()
    if bundle.get("feature_mode") == ROI_PIXEL_TEXTURE_FEATURE_MODE:
        features = extract_roi_pixel_texture_features(
            [image_path],
            masks=[defect_mask],
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
        )
    elif bundle.get("feature_mode") == ROI_SHAPE_TEXTURE_FEATURE_MODE:
        features = extract_roi_shape_texture_features(
            [image_path],
            masks=[defect_mask],
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
        )
    elif bundle.get("feature_mode") == ROI_TEXTURE_FEATURE_MODE:
        features = extract_roi_texture_features(
            [image_path],
            masks=[defect_mask],
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
        )
    elif bundle.get("feature_mode") == GLOBAL_TEXTURE_FEATURE_MODE:
        features = extract_global_texture_features(
            [image_path],
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            device=device,
        )
    else:
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
        "confidence": round(float(max(probabilities)), 4),
        "class_probabilities": class_probabilities,
    }
