import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MVTEC_DATASET_ROOT = Path(
    os.getenv("MVTEC_DATASET_ROOT", str(DATA_ROOT / "raw" / "mvtec_anomaly_detection"))
).expanduser()
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CATEGORY = os.getenv("MVTEC_CATEGORY", "bottle").strip().lower().replace("-", "_").replace(" ", "_")


def category_data_dir(category: str = DEFAULT_CATEGORY) -> Path:
    return MVTEC_DATASET_ROOT / category


def category_checkpoint_path(category: str = DEFAULT_CATEGORY, model_kind: str = "padim") -> Path:
    if category == "bottle":
        return MODELS_DIR / "local_checkpoints" / "padim_mvtec_bottle_v1.ckpt"
    return MODELS_DIR / "categories" / category / f"{model_kind}_v1.ckpt"


RAW_DATA_DIR = category_data_dir(DEFAULT_CATEGORY)
MODEL_CHECKPOINT_PATH = Path(
    os.getenv("MODEL_CHECKPOINT_PATH", str(category_checkpoint_path(DEFAULT_CATEGORY)))
).expanduser()
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

IMAGE_SIZE = (256, 256)
