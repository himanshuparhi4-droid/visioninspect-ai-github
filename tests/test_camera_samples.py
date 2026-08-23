import sys
from pathlib import Path

import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.routes import inspection_routes  # noqa: E402


def test_bundled_camera_demo_samples_are_available_without_dataset(monkeypatch, tmp_path):
    labels = {"good", "broken_large", "broken_small", "contamination"}
    monkeypatch.setattr(
        inspection_routes,
        "get_camera_sample_root",
        lambda category: tmp_path / "missing-dataset" / category,
    )

    for label in labels:
        paths = inspection_routes.camera_sample_paths("bottle", label)
        assert len(paths) >= 3
        assert all(path.suffix == ".png" for path in paths)


def test_bundled_bottle_samples_match_the_deployed_openvino_pipeline():
    pytest.importorskip("openvino")

    from ml.inference import InferenceConfig, inspect_image
    from ml.model_registry import category_model_spec

    spec = category_model_spec("bottle")
    if spec.openvino_path is None or spec.openvino_calibrator_path is None:
        pytest.skip("Bottle OpenVINO runtime is not configured")
    if not spec.openvino_path.exists() or not spec.openvino_calibrator_path.exists():
        pytest.skip("Bottle OpenVINO artifacts are not available")

    config = InferenceConfig(
        category=spec.category,
        anomaly_model_kind=spec.model_kind,
        use_padim_inference=False,
        use_openvino_inference=True,
        openvino_inference_device="CPU",
        padim_inference_accelerator="cpu",
        model_checkpoint_path=spec.checkpoint_path,
        classifier_model_path=spec.classifier_path,
        cnn_classifier_model_path=spec.cnn_classifier_path,
        model_metadata_path=spec.metadata_path,
        baseline_profile_path=spec.baseline_profile_path,
        baseline_threshold=spec.baseline_score_threshold,
        baseline_residual_threshold=spec.baseline_residual_threshold,
        padim_score_threshold=spec.padim_score_threshold,
        review_severity_threshold=40.0,
        fail_severity_threshold=60.0,
        subtype_confidence_threshold=spec.subtype_confidence_threshold,
        openvino_path=spec.openvino_path,
        openvino_calibrator_path=spec.openvino_calibrator_path,
        input_size=spec.input_size,
    )

    sample_root = BACKEND_DIR / "app" / "demo_samples" / "bottle" / "test"
    for expected_label in ("good", "broken_large", "broken_small", "contamination"):
        for image_path in sorted((sample_root / expected_label).glob("*.png")):
            image = cv2.imread(str(image_path))
            assert image is not None
            assert image.shape == (spec.input_size, spec.input_size, 3)

            result = inspect_image(image_path, config)

            assert result["prediction"] == ("Good" if expected_label == "good" else "Defective"), (
                f"{expected_label}/{image_path.name} was classified incorrectly: "
                f"score={result['anomaly_score']}, "
                f"probability={result['explainability']['calibrated_defect_probability']}"
            )
            assert result["defect_type"] == expected_label
            assert result["pass_fail"] == ("Pass" if expected_label == "good" else "Fail")
            assert result["active_inference_engine"] == "padim_openvino"
            assert result["fallback_used"] is False
