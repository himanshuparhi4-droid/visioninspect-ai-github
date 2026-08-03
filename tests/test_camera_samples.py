import sys
from pathlib import Path

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
