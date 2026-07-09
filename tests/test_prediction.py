from pathlib import Path

from ml import predict


def test_inspect_image_returns_backend_ready_output(monkeypatch):
    expected_image = Path("sample.png")

    def fake_inspect_image_file(image_path):
        assert image_path == expected_image
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
            "model_version": "padim:v1 (padim)",
        }

    monkeypatch.setattr("app.services.prediction_service.inspect_image_file", fake_inspect_image_file)

    result = predict.inspect_image(expected_image)

    assert result == {
        "input_image": str(expected_image),
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
    }
