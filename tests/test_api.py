import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.models.user_model import User  # noqa: E402
from app.routes.inspection_routes import automatic_metadata  # noqa: E402
from app.routes.rework_routes import ensure_resolution_notes  # noqa: E402
from app.services import cloudinary_service  # noqa: E402
from app.services.cloudinary_service import CloudStorageError  # noqa: E402
from app.services.model_settings_service import build_model_metrics_payload  # noqa: E402
from main import app  # noqa: E402


def test_health_endpoint_reports_model_artifacts():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert response.headers["x-content-type-options"] == "nosniff"
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "VisionInspect AI"
    assert payload["artifacts"]["defect_classifier"] is True
    assert payload["artifacts"]["baseline_profile"] is True
    assert payload["artifacts"]["portable_categories"] == 15


def test_health_live_and_ready_endpoints_are_exposed():
    with TestClient(app) as client:
        live_response = client.get("/health/live")
        ready_response = client.get("/health/ready")

    assert live_response.status_code == 200
    assert live_response.json()["status"] == "alive"
    assert ready_response.status_code in {200, 503}
    assert "checks" in ready_response.json()


def test_api_errors_include_request_id_and_message():
    with TestClient(app) as client:
        response = client.get("/users")

    assert response.status_code == 401
    payload = response.json()
    assert payload["error"]["message"]
    assert payload["error"]["request_id"]
    assert response.headers["x-request-id"] == payload["error"]["request_id"]


def test_openapi_exposes_core_platform_routes():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    paths = response.json()["paths"]
    assert "/auth/register" in paths
    assert "/auth/login" in paths
    assert "/users" in paths
    assert "/inspections/inspect" in paths
    assert "/inspections/batch-inspect" in paths
    assert "/inspections/{inspection_id}/metadata" in paths
    assert "/analytics/summary" in paths
    assert "/reports/inspection/{inspection_id}" in paths
    assert "/rework/tickets" in paths
    assert "/production/catalog" in paths
    assert "/model/metrics" in paths
    assert "/model/settings" in paths
    assert "/health/live" in paths
    assert "/health/ready" in paths


def test_model_metrics_include_every_category_baseline():
    payload = build_model_metrics_payload()
    rows = payload["baseline_metrics"]
    category_rows = payload["category_models"]

    assert len(rows) == 15
    assert {row["category"] for row in rows} == {
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    }
    for row in rows:
        assert row["samples"]
        assert row["threshold"] is not None
        assert row["balanced_accuracy"] is not None
        assert row["f1"] is not None
        assert row["auroc"] is not None

    assert len(category_rows) == 15
    assert all(row["image_f1"] is not None for row in category_rows)


def test_automatic_metadata_fills_blanks_without_overwriting_values():
    user = User.model_construct(name="Quality Engineer", email="engineer@example.com", hashed_password="hashed")

    generated = automatic_metadata({"product_id": "BOTTLE-CUSTOM"}, user, "line-image.png")

    assert generated["product_id"] == "BOTTLE-CUSTOM"
    assert generated["batch_number"].startswith("AUTO-")
    assert generated["production_line"] == "Line-Manual-01"
    assert generated["shift"] == "Auto Shift"
    assert generated["operator_name"] == "Quality Engineer"
    assert generated["source_label"] == "line-image.png"


def test_cloud_storage_failure_has_actionable_message(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image")
    monkeypatch.setattr(cloudinary_service, "cloudinary_is_configured", lambda: True)
    monkeypatch.setattr(cloudinary_service.settings, "environment", "production")

    def fail_upload(*args, **kwargs):
        raise TimeoutError("storage timeout")

    monkeypatch.setattr(cloudinary_service.cloudinary.uploader, "upload", fail_upload)

    with pytest.raises(CloudStorageError, match="20 seconds"):
        cloudinary_service.upload_image_or_local_url(image_path, "original")


def test_rework_completion_requires_resolution_notes():
    with pytest.raises(HTTPException) as exc_info:
        ensure_resolution_notes("completed", " ")

    assert getattr(exc_info.value, "status_code", None) == 422
    ensure_resolution_notes("completed", "Replaced the damaged component and re-inspected the product.")
