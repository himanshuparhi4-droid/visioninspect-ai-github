from pathlib import Path
from threading import Barrier

import cv2
import numpy as np

from ml import predict
from ml.inference import (
    InferenceConfig,
    build_explainability,
    calibrated_subtype_confidence,
    classify_prediction,
    compute_defect_geometry,
)
from ml.model_registry import category_model_spec
from ml.padim_detector import openvino_spatial_features


def test_inspect_image_returns_backend_ready_output(monkeypatch):
    expected_image = Path("sample.png")

    def fake_runtime_settings():
        class RuntimeSettings:
            baseline_threshold = 1.34
            padim_score_threshold = 0.5
            review_severity_threshold = 40
            fail_severity_threshold = 60

        return RuntimeSettings()

    def fake_inspect_image_runtime(image_path, config):
        assert image_path == expected_image
        assert config.baseline_threshold == category_model_spec("bottle").baseline_score_threshold
        return {
            "prediction": "Defective",
            "defect_type": "contamination",
            "confidence": 0.91,
            "anomaly_score": 0.72,
            "defect_area_ratio": 0.08,
            "heatmap_path": "inline:notebook_heatmap",
            "processed_image_path": "inline:processed_image",
            "severity_score": 82.5,
            "severity_level": "Critical",
            "pass_fail": "Fail",
            "recommended_action": "Reject product or send to rework based on QA policy",
            "model_used": "padim:v1 (padim)",
            "active_inference_engine": "padim",
            "fallback_used": False,
            "fallback_reason": None,
        }

    monkeypatch.setattr("app.services.model_settings_service.load_runtime_settings", fake_runtime_settings)
    monkeypatch.setattr("ml.inference.inspect_image", fake_inspect_image_runtime)

    result = predict.inspect_image(expected_image)

    assert result == {
        "input_image": str(expected_image),
        "category": "bottle",
        "prediction": "Defective",
        "defect_type": "contamination",
        "confidence": 0.91,
        "anomaly_score": 0.72,
        "defect_area_ratio": 0.08,
        "heatmap_path": "inline:not_saved_by_cli",
        "processed_image_path": "inline:not_saved_by_cli",
        "severity_score": 82.5,
        "severity_level": "Critical",
        "pass_fail": "Fail",
        "recommended_action": "Reject product or send to rework based on QA policy",
        "model_used": "padim:v1 (padim)",
        "active_inference_engine": "padim",
        "fallback_used": False,
        "fallback_reason": None,
    }


def test_runtime_keeps_detection_and_subtype_confidence_separate(monkeypatch, tmp_path):
    from ml.inference import inspect_image

    runtime_events = []
    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), np.full((32, 32, 3), 128, dtype=np.uint8))
    anomaly_map = np.zeros((32, 32), dtype=np.float32)
    anomaly_map[12:20, 12:20] = 1.0
    pred_mask = anomaly_map > 0.5
    global_features = np.ones((1, 512), dtype=np.float32)

    monkeypatch.setattr(
        "ml.inference.live_anomaly_prediction",
        lambda *_: {
            "engine": "padim_openvino",
            "detector_kind": "normal-memory",
            "anomaly_score": 0.8,
            "decision_threshold": 0.5,
            "is_defective": True,
            "detection_confidence": 0.8,
            "anomaly_map": anomaly_map,
            "pred_mask": pred_mask,
            "global_features": global_features,
            "fallback_used": False,
            "fallback_reason": None,
        },
    )

    def fake_classify_prediction(*_args, global_features=None):
        assert runtime_events == ["detector_released"]
        assert global_features is not None
        assert np.array_equal(global_features, np.ones((1, 512), dtype=np.float32))
        return {
            "defect_type": "contamination",
            "confidence": 0.6,
            "classification_confidence": 0.6,
            "class_probabilities": {"contamination": 0.6},
        }

    monkeypatch.setattr(
        "ml.inference.classify_prediction",
        fake_classify_prediction,
    )
    monkeypatch.setattr(
        "ml.inference.release_anomaly_runtimes",
        lambda: runtime_events.append("detector_released"),
    )

    missing = tmp_path / "missing"
    config = InferenceConfig(
        category="bottle",
        anomaly_model_kind="padim",
        use_padim_inference=False,
        padim_inference_accelerator="cpu",
        model_checkpoint_path=missing,
        classifier_model_path=missing,
        model_metadata_path=missing,
        baseline_profile_path=missing,
        baseline_threshold=0.5,
        baseline_residual_threshold=0.5,
        padim_score_threshold=0.5,
        review_severity_threshold=40,
        fail_severity_threshold=60,
        subtype_model_macro_f1=0.70,
        release_detector_before_classification=True,
    )

    result = inspect_image(image_path, config)

    assert result["confidence"] == 0.8
    assert result["detection_confidence"] == 0.8
    assert result["classification_confidence"] == 0.6
    assert result["severity_components"]["confidence_score"] == 80.0
    assert result["explainability"]["detection_confidence"] == 0.8
    assert result["explainability"]["classification_confidence"] == 0.6
    assert result["subtype_model_status"] == "Manual review"
    assert result["manual_review_required"] is True


def test_saved_isotonic_mapping_calibrates_subtype_confidence():
    calibration = {
        "method": "isotonic_oof_top_label",
        "x_thresholds": [0.4, 0.6, 0.9],
        "y_thresholds": [0.3, 0.7, 0.95],
    }

    calibrated, applied = calibrated_subtype_confidence(0.75, calibration)

    assert applied is True
    assert np.isclose(calibrated, 0.825)


def test_unverified_calibration_keeps_raw_subtype_confidence():
    calibrated, applied = calibrated_subtype_confidence(
        0.72,
        {"method": "not_applied_no_ece_gain"},
    )

    assert applied is False
    assert calibrated == 0.72


def test_geometry_uses_product_configured_critical_zones():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1

    critical = compute_defect_geometry(mask, "crack", ("center",))
    cosmetic = compute_defect_geometry(mask, "crack", ("edge",))

    assert critical["is_critical_location"] is True
    assert "center" in critical["detected_regions"]
    assert cosmetic["is_critical_location"] is False


def test_missing_classifier_reports_unknown_without_inventing_a_class(tmp_path):
    missing = tmp_path / "missing"
    config = InferenceConfig(
        category="bottle",
        anomaly_model_kind="padim",
        use_padim_inference=False,
        padim_inference_accelerator="cpu",
        model_checkpoint_path=missing,
        classifier_model_path=missing,
        model_metadata_path=missing,
        baseline_profile_path=missing,
        baseline_threshold=0.5,
        baseline_residual_threshold=0.5,
        padim_score_threshold=0.5,
        review_severity_threshold=40,
        fail_severity_threshold=60,
    )

    classification = classify_prediction(
        tmp_path / "sample.png",
        score=0.8,
        detection_confidence=0.82,
        is_defective=True,
        binary_mask=np.ones((8, 8), dtype=np.uint8),
        config=config,
    )

    assert classification["defect_type"] == "unknown_defect"
    assert classification["classification_confidence"] is None
    assert "not found" in classification["classification_error"]


def test_visual_outputs_upload_concurrently(monkeypatch, tmp_path):
    from app.services.prediction_service import save_visual_outputs

    barrier = Barrier(2)

    def temporary_uploads_path(folder):
        path = tmp_path / folder
        path.mkdir(parents=True, exist_ok=True)
        return path

    def synchronized_upload(_path, folder):
        barrier.wait(timeout=2)
        return f"https://storage.example/{folder}.png"

    monkeypatch.setattr("app.services.prediction_service.uploads_path", temporary_uploads_path)
    monkeypatch.setattr("app.services.prediction_service.upload_image_or_local_url", synchronized_upload)

    outputs = save_visual_outputs(
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.uint8),
    )

    assert outputs["processed_image_url"].endswith("processed.png")
    assert outputs["heatmap_url"].endswith("heatmaps.png")


def test_openvino_spatial_features_are_fixed_length_and_finite():
    anomaly_map = np.linspace(0.1, 0.9, 256 * 256, dtype=np.float32).reshape(256, 256)

    features = openvino_spatial_features(0.63, anomaly_map)

    assert features.shape == (72,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_explainability_describes_spatially_calibrated_decision():
    explanation = build_explainability(
        prediction="Defective",
        defect_type="poke_insulation",
        confidence=0.91,
        detection_confidence=0.88,
        classification_confidence=0.91,
        classification_error=None,
        anomaly_score_value=0.47,
        decision_threshold=0.52,
        geometry={
            "area_ratio": 0.01,
            "is_critical_location": False,
            "critical_zones": [],
            "detected_regions": [],
            "defect_center_x_ratio": 0.5,
            "defect_center_y_ratio": 0.5,
        },
        severity={"components": {}},
        anomaly_map_value=np.ones((16, 16), dtype=np.float32),
        engine="patchcore_openvino",
        fallback_used=False,
        fallback_reason=None,
        decision_basis="spatial_calibrator",
        calibrated_defect_probability=0.87,
        calibration_threshold=0.5,
    )

    assert explanation["decision_basis"] == "spatial_calibrator"
    assert explanation["calibrated_defect_probability"] == 0.87
    assert "87.0% defect probability" in explanation["notes"][0]
