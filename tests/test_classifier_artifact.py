from pathlib import Path

import numpy as np

from ml.classifier import (
    GLOBAL_TEXTURE_FEATURE_MODE,
    ROI_PIXEL_TEXTURE_FEATURE_MODE,
    ROI_SHAPE_TEXTURE_FEATURE_MODE,
    ROI_TEXTURE_FEATURE_MODE,
    build_opencv_resnet18_feature_extractor,
    extract_features,
    load_classifier_bundle,
)
from ml.defect_classifier import classify_defect_type


def test_defect_classifier_artifact_loads():
    path = Path("models/defect_classifier.pkl")

    assert path.exists()

    bundle = load_classifier_bundle(path)
    assert "classifier" in bundle
    assert set(bundle["labels"]) == {"broken_large", "broken_small", "contamination"}
    assert bundle["defect_only"] is True
    assert bundle["feature_mode"] in {
        GLOBAL_TEXTURE_FEATURE_MODE,
        ROI_TEXTURE_FEATURE_MODE,
        ROI_SHAPE_TEXTURE_FEATURE_MODE,
        ROI_PIXEL_TEXTURE_FEATURE_MODE,
    }
    assert bundle["metrics"]["macro_f1"] >= 0.8


def test_opencv_feature_runtime_returns_resnet_embeddings():
    image_path = Path("models/inference/normal_reference.png")
    runtime, preprocess, device = build_opencv_resnet18_feature_extractor()

    features = extract_features(
        [image_path],
        feature_extractor=runtime,
        preprocess=preprocess,
        device=device,
    )

    assert features.shape == (1, 512)
    assert np.isfinite(features).all()


def test_classifier_prefers_promoted_cnn_artifact(monkeypatch, tmp_path):
    classifier_path = tmp_path / "defect_classifier.pkl"
    classifier_path.write_bytes(b"placeholder")
    cnn_path = tmp_path / "cnn_defect_classifier.pt"
    cnn_path.write_bytes(b"placeholder")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")

    def fake_cnn_predict(image_path_arg, artifact_path_arg, *, defect_mask=None):
        assert Path(image_path_arg) == image_path
        assert Path(artifact_path_arg) == cnn_path
        assert defect_mask is not None
        return {
            "defect_type": "crack",
            "confidence": 0.91,
            "class_probabilities": {"crack": 0.91},
            "classifier_engine": "fine_tuned_resnet18",
        }

    monkeypatch.setattr("ml.cnn_classifier.predict_cnn_defect_type", fake_cnn_predict)

    result = classify_defect_type(
        image_path,
        classifier_path,
        defect_mask=np.ones((8, 8), dtype=bool),
    )

    assert result["defect_type"] == "crack"
    assert result["classifier_engine"] == "fine_tuned_resnet18"
