from pathlib import Path

from ml.classifier import LABEL_ORDER, load_classifier_bundle


def test_defect_classifier_artifact_loads():
    path = Path("models/defect_classifier.pkl")

    assert path.exists()

    bundle = load_classifier_bundle(path)
    assert "classifier" in bundle
    assert bundle["labels"] == LABEL_ORDER
    assert bundle["feature_extractor"] == "resnet18_imagenet1k_v1"
    assert bundle["metrics"]["accuracy"] >= 0.9
