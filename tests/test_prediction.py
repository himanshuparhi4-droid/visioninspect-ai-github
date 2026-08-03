from pathlib import Path

import cv2
import numpy as np

from ml import predict
from ml.inference import InferenceConfig, classify_prediction, compute_defect_geometry
from ml.model_registry import category_model_spec


def test_inspect_image_returns_backend_ready_output(monkeypatch):
    expected_image = Path("sample.png")

    def fake_runtime_settings():
        class RuntimeSettings:
            baseline_threshold = 1.34
            padim_score_threshold = 0.5
            review_severity_threshold = 40
            fail_severity_threshold = 60

        return RuntimeSettings()

    def fake_resolve_backend_path(value):
        return Path(value)

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
    monkeypatch.setattr("app.services.prediction_service.resolve_backend_path", fake_resolve_backend_path)
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

    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), np.full((32, 32, 3), 128, dtype=np.uint8))
    anomaly_map = np.zeros((32, 32), dtype=np.float32)
    anomaly_map[12:20, 12:20] = 1.0
    pred_mask = anomaly_map > 0.5

    monkeypatch.setattr(
        "ml.inference.live_anomaly_prediction",
        lambda *_: {
            "engine": "baseline",
            "detector_kind": "normal-memory",
            "anomaly_score": 0.8,
            "decision_threshold": 0.5,
            "is_defective": True,
            "detection_confidence": 0.8,
            "anomaly_map": anomaly_map,
            "pred_mask": pred_mask,
            "fallback_used": False,
            "fallback_reason": None,
        },
    )
    monkeypatch.setattr(
        "ml.inference.classify_prediction",
        lambda *_: {
            "defect_type": "contamination",
            "confidence": 0.6,
            "classification_confidence": 0.6,
            "class_probabilities": {"contamination": 0.6},
        },
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
    )

    result = inspect_image(image_path, config)

    assert result["confidence"] == 0.6
    assert result["detection_confidence"] == 0.8
    assert result["classification_confidence"] == 0.6
    assert result["severity_components"]["confidence_score"] == 80.0
    assert result["explainability"]["detection_confidence"] == 0.8
    assert result["explainability"]["classification_confidence"] == 0.6


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
