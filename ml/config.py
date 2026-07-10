from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw" / "mvtec_anomaly_detection" / "bottle"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CHECKPOINT_PATH = MODELS_DIR / "checkpoints" / "padim_mvtec_bottle_v1.ckpt"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

IMAGE_SIZE = (256, 256)
DEFECT_LABELS = ["good", "broken_large", "broken_small", "contamination"]
