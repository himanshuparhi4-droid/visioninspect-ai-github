from pathlib import Path

import numpy as np

from ml.classifier import (
    GLOBAL_TEXTURE_FEATURE_MODE,
    ROI_PIXEL_TEXTURE_FEATURE_MODE,
    ROI_SHAPE_TEXTURE_FEATURE_MODE,
    ROI_TEXTURE_FEATURE_MODE,
    build_opencv_resnet18_feature_extractor,
    export_portable_forest,
    extract_features,
    load_classifier_bundle,
    predict_portable_forest,
)
from ml.defect_classifier import classify_defect_type, predict_portable_cnn_defect_type


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


def test_classifier_prefers_portable_cnn_artifact(monkeypatch, tmp_path):
    classifier_path = tmp_path / "defect_classifier.pkl"
    classifier_path.write_bytes(b"placeholder")
    cnn_path = tmp_path / "cnn_defect_classifier.onnx"
    cnn_path.write_bytes(b"placeholder")
    metadata_path = tmp_path / "cnn_defect_classifier.json"
    metadata_path.write_text("{}", encoding="utf-8")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")

    def fake_cnn_predict(image_path_arg, artifact_path_arg, metadata_path_arg, *, defect_mask=None):
        assert Path(image_path_arg) == image_path
        assert Path(artifact_path_arg) == cnn_path
        assert Path(metadata_path_arg) == metadata_path
        assert defect_mask is not None
        return {
            "defect_type": "crack",
            "confidence": 0.91,
            "class_probabilities": {"crack": 0.91},
            "classifier_engine": "fine_tuned_resnet18_onnx",
        }

    monkeypatch.setattr("ml.defect_classifier.predict_portable_cnn_defect_type", fake_cnn_predict)

    result = classify_defect_type(
        image_path,
        classifier_path,
        defect_mask=np.ones((8, 8), dtype=bool),
    )

    assert result["defect_type"] == "crack"
    assert result["classifier_engine"] == "fine_tuned_resnet18_onnx"


def test_portable_cnn_artifacts_run_without_pytorch():
    image_path = Path("models/inference/normal_reference.png")
    for category in ("capsule", "wood"):
        model_dir = Path("models/categories") / category
        result = predict_portable_cnn_defect_type(
            image_path,
            model_dir / "cnn_defect_classifier.onnx",
            model_dir / "cnn_defect_classifier.json",
        )

        assert result["defect_type"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["classifier_engine"] == "fine_tuned_resnet18_onnx"


def test_compact_classifier_reuses_precomputed_global_features(monkeypatch, tmp_path):
    import cv2

    classifier_path = tmp_path / "defect_classifier.pkl"
    classifier_path.touch()
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.full((32, 32, 3), 128, dtype=np.uint8))

    class FakeClassifier:
        classes_ = np.asarray(["crack"])

        @staticmethod
        def predict(features):
            assert features.shape[0] == 1
            return np.asarray(["crack"])

        @staticmethod
        def predict_proba(features):
            assert features.shape[0] == 1
            return np.asarray([[1.0]])

    monkeypatch.setattr(
        "ml.defect_classifier.load_classifier_runtime",
        lambda _path: {
            "classifier": FakeClassifier(),
            "feature_mode": GLOBAL_TEXTURE_FEATURE_MODE,
        },
    )
    monkeypatch.setattr("ml.defect_classifier.shared_feature_runtime", lambda: (object(), object(), "cpu"))
    monkeypatch.setattr(
        "ml.classifier.extract_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("global features were recomputed")),
    )

    result = classify_defect_type(
        image_path,
        classifier_path,
        global_features=np.ones((1, 512), dtype=np.float32),
    )

    assert result["defect_type"] == "crack"


def test_portable_forest_matches_scaled_sklearn_pipeline(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(
        [[0.0, 0.1], [0.2, 0.0], [0.8, 0.9], [1.0, 0.8], [0.1, 0.3], [0.9, 1.0]],
        dtype=np.float32,
    )
    labels = np.asarray(["good", "good", "defect", "defect", "good", "defect"])
    classifier = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=20, random_state=42),
    ).fit(features, labels)
    artifact_path = tmp_path / "portable_forest.npz"
    export_portable_forest(classifier, artifact_path, feature_mode="test")

    expected = classifier.predict_proba(features)
    actual = predict_portable_forest(artifact_path, features)

    assert list(actual["classes"]) == list(classifier.classes_)
    assert np.allclose(actual["probabilities"], expected, atol=1e-7)
    assert list(actual["labels"]) == list(classifier.predict(features))
