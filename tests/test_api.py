import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from main import app  # noqa: E402


def test_health_endpoint_reports_model_artifacts():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["artifacts"]["defect_classifier"] is True
    assert payload["artifacts"]["baseline_reference"] is True


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
