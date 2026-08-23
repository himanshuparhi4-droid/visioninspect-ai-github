from pathlib import Path

import joblib
import numpy as np
from app.models.inspection_model import Inspection
from app.models.production_model import BatchRecord
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.baseline_detector import load_reference_profile
from ml.classifier import export_portable_forest, load_portable_forest, predict_portable_forest
from ml.model_registry import (
    CATEGORY_DEFECT_LABELS,
    SUPPORTED_CATEGORIES,
    CategoryModelSpec,
    category_model_spec,
    category_model_statuses,
    classifier_runtime_status,
    openvino_runtime_is_memory_safe,
)
from scripts.calibrate_category_thresholds import sync_registry_calibration


def portable_spec(tmp_path: Path) -> CategoryModelSpec:
    profile = tmp_path / "normal_profile.npz"
    metadata = tmp_path / "model_metadata.json"
    classifier = tmp_path / "defect_classifier.pkl"
    profile.touch()
    metadata.write_text("{}", encoding="utf-8")
    classifier.touch()
    return CategoryModelSpec(
        category="bottle",
        model_kind="padim",
        checkpoint_path=tmp_path / "missing.ckpt",
        classifier_path=classifier,
        baseline_profile_path=profile,
        metadata_path=metadata,
    )


def test_compact_artifacts_are_runnable_without_advanced_checkpoint(tmp_path):
    spec = portable_spec(tmp_path)

    assert spec.is_runnable is True
    assert spec.has_advanced_model is False
    assert spec.is_trained is False


def test_quality_records_do_not_expire_automatically():
    for model in (Inspection, BatchRecord):
        indexes = model.Settings.indexes
        assert all(getattr(index, "document", {}).get("expireAfterSeconds") is None for index in indexes)


def test_every_supported_category_has_portable_runtime_artifacts():
    statuses = category_model_statuses()

    assert {item["category"] for item in statuses} == set(SUPPORTED_CATEGORIES)
    assert all(item["available"] for item in statuses)
    assert all(item["classification_trained"] for item in statuses)
    assert all(
        len(item["artifacts"][artifact]["sha256"]) == 64
        for item in statuses
        for artifact in ("profile", "classifier", "metadata")
    )

    for category in SUPPORTED_CATEGORIES:
        spec = category_model_spec(category)
        profile = load_reference_profile(spec.baseline_profile_path)
        classifier = joblib.load(spec.classifier_path)
        assert profile["mean"].shape == profile["std"].shape == profile["foreground_mask"].shape
        assert profile["embedding_bank"].ndim == 2
        assert profile["embedding_bank"].shape[1] == 512
        assert "classifier" in classifier
        assert classifier["defect_only"] is True
        assert "good" not in classifier["labels"]
        assert set(classifier["labels"]) == set(CATEGORY_DEFECT_LABELS[category])


def test_registry_can_explicitly_disable_an_unused_cnn_classifier():
    capsule = category_model_spec("capsule")

    assert capsule.cnn_classifier_path is None
    assert classifier_runtime_status(capsule)["engine"] == "sklearn_feature_classifier"


def test_low_memory_profile_keeps_only_memory_safe_openvino_pairs():
    bottle = category_model_spec("bottle")
    cable = category_model_spec("cable")
    pill = category_model_spec("pill")

    assert openvino_runtime_is_memory_safe(bottle, "fine_tuned_resnet18_onnx") is True
    assert openvino_runtime_is_memory_safe(cable, "portable_forest") is True
    assert openvino_runtime_is_memory_safe(pill, "sklearn_feature_classifier") is False

    statuses = {
        item["category"]: item
        for item in category_model_statuses(
            advanced_enabled=True,
            openvino_enabled=True,
            resource_constrained=True,
        )
    }
    assert statuses["bottle"]["active_engine"] == "padim_openvino"
    assert statuses["cable"]["active_engine"] == "patchcore_openvino"
    assert statuses["pill"]["active_engine"] == "portable_baseline"
    assert statuses["pill"]["openvino_available"] is True
    assert statuses["pill"]["openvino_deferred_for_memory"] is True


def test_registered_portable_detector_calibrators_are_complete():
    calibrated_categories = {"capsule", "grid", "hazelnut", "pill", "screw", "transistor"}

    for category in calibrated_categories:
        spec = category_model_spec(category)
        assert spec.portable_detector_calibrator_path is not None
        assert spec.portable_detector_calibrator_path.exists()

        profile = load_reference_profile(spec.baseline_profile_path)
        runtime = load_portable_forest(
            str(spec.portable_detector_calibrator_path),
            spec.portable_detector_calibrator_path.stat().st_mtime_ns,
        )
        assert profile["spatial_embedding_bank"].shape == (13, 128, 512)
        assert str(runtime["feature_mode"]) == "portable_anomaly_spatial_v1"
        assert 0.0 < float(runtime["decision_threshold"]) < 1.0


def test_render_requirements_include_portable_inference_dependencies():
    requirements = (Path(__file__).resolve().parents[1] / "backend" / "requirements.txt").read_text(encoding="utf-8")

    assert "opencv-python-headless==5.0.0.93" in requirements
    assert 'torch==2.11.0; sys_platform != "linux"' in requirements
    assert 'torchvision==0.26.0; sys_platform != "linux"' in requirements
    assert 'torch==2.11.0+cpu; sys_platform == "linux"' not in requirements
    assert "pandas==" not in requirements


def test_shared_openvino_feature_model_is_github_safe():
    model_path = Path("models/shared/resnet18_features_fp16.xml")
    weights_path = model_path.with_suffix(".bin")
    pytorch_weights_path = Path("models/shared/resnet18-f37072fd.pth")

    assert model_path.exists()
    assert weights_path.exists()
    assert 1_000_000 < weights_path.stat().st_size < 50_000_000
    assert not pytorch_weights_path.exists()



def test_portable_logistic_runtime_matches_sklearn(tmp_path):
    features = np.asarray(
        [[-2.0, -1.0], [-1.0, -0.5], [1.0, 0.5], [2.0, 1.0]],
        dtype=np.float32,
    )
    targets = np.asarray([0, 0, 1, 1])
    classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=42)).fit(features, targets)
    artifact = tmp_path / "calibrator.npz"

    export_portable_forest(
        classifier,
        artifact,
        feature_mode="openvino_spatial_v1",
        decision_threshold=0.4,
    )
    result = predict_portable_forest(artifact, features)

    assert np.allclose(result["probabilities"], classifier.predict_proba(features), atol=1e-6)
    assert result["decision_threshold"] == np.float32(0.4)


def test_legacy_baseline_calibration_can_sync_without_a_calibrator(tmp_path):
    spec = portable_spec(tmp_path)
    registry_entry = {"portable_detector_calibrator_path": "models/stale.npz"}

    sync_registry_calibration(
        registry_entry,
        "baseline",
        {
            "threshold": 0.010643,
            "residual_threshold": 0.790813,
            "cv_balanced_accuracy": 0.9309,
            "cv_f1": 0.9524,
        },
        spec,
    )

    assert registry_entry["baseline_score_threshold"] == 0.010643
    assert registry_entry["baseline_residual_threshold"] == 0.790813
    assert "portable_detector_calibrator_path" not in registry_entry
