from pathlib import Path

import cv2
import pandas as pd

from ml.config import RAW_DATA_DIR

DEFECT_LABELS = ["broken_large", "broken_small", "contamination"]
ALL_LABELS = ["good", *DEFECT_LABELS]
DATASET_COLUMNS = [
    "split",
    "label",
    "target",
    "target_name",
    "is_defective",
    "image_path",
    "mask_path",
    "width",
    "height",
    "channels",
]


def read_image_shape(image_path: Path) -> tuple[int | None, int | None, int | None]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, None
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return width, height, channels


def get_mask_path(root: Path, image_path: Path, label: str) -> Path | None:
    if label == "good":
        return None
    mask_path = root / "ground_truth" / label / f"{image_path.stem}_mask.png"
    return mask_path if mask_path.exists() else None


def collect_bottle_images(root: Path = RAW_DATA_DIR) -> list[dict]:
    """Collect MVTec bottle image paths with labels and optional masks."""
    records: list[dict] = []

    train_good = root / "train" / "good"
    for image_path in sorted(train_good.glob("*.png")) if train_good.exists() else []:
        width, height, channels = read_image_shape(image_path)
        records.append(
            {
                "split": "train",
                "label": "good",
                "target": 0,
                "target_name": "good",
                "is_defective": False,
                "image_path": str(image_path),
                "mask_path": None,
                "width": width,
                "height": height,
                "channels": channels,
            }
        )

    test_dir = root / "test"
    for label in ALL_LABELS:
        label_dir = test_dir / label
        for image_path in sorted(label_dir.glob("*.png")) if label_dir.exists() else []:
            is_defective = label != "good"
            width, height, channels = read_image_shape(image_path)
            mask_path = get_mask_path(root, image_path, label)
            records.append(
                {
                    "split": "test",
                    "label": label,
                    "target": 1 if is_defective else 0,
                    "target_name": "defective" if is_defective else "good",
                    "is_defective": is_defective,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if mask_path else None,
                    "width": width,
                    "height": height,
                    "channels": channels,
                }
            )

    return records


def load_bottle_dataframe(root: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return pd.DataFrame(collect_bottle_images(root), columns=DATASET_COLUMNS)
