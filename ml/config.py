import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def default_mvtec_dataset_root() -> Path:
    configured = os.getenv("MVTEC_DATASET_ROOT")
    if configured:
        return Path(configured).expanduser()
    candidates = (
        DATA_ROOT / "raw" / "mvtec_anomaly_detection",
        DATA_ROOT / "raw" / "mvtec_anomaly_detection1",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


MVTEC_DATASET_ROOT = default_mvtec_dataset_root()
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CATEGORY = os.getenv("MVTEC_CATEGORY", "bottle").strip().lower().replace("-", "_").replace(" ", "_")


def category_data_dir(category: str = DEFAULT_CATEGORY) -> Path:
    return MVTEC_DATASET_ROOT / category


RAW_DATA_DIR = category_data_dir(DEFAULT_CATEGORY)

IMAGE_SIZE = (256, 256)
