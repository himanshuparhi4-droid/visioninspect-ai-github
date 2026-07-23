from pathlib import Path

from ml import predict


def test_inspect_image_returns_backend_ready_output(monkeypatch):
    expected_image = Path("sample.png")

    def fake_runtime_settings():
        class RuntimeSettings:
            baseline_threshold = 1.45
            padim_score_threshold = 0.5
            review_severity_threshold = 40
            fail_severity_threshold = 60

        return RuntimeSettings()

    def fake_resolve_backend_path(value):
        return Path(value)

    def fake_inspect_image_runtime(image_path, config):
        assert image_path == expected_image
        assert config.baseline_threshold == 1.45
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
