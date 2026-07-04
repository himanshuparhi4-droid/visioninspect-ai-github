import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.routes.inspection_routes import camera_sample_paths  # noqa: E402


def test_bundled_camera_demo_samples_are_available():
    labels = {"good", "broken_large", "broken_small", "contamination"}

    for label in labels:
        paths = camera_sample_paths(label)
        assert len(paths) >= 3
        assert all(path.suffix == ".png" for path in paths)
