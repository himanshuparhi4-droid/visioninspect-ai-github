import cv2
import numpy as np


def image_metadata(image: np.ndarray) -> dict:
    """Return basic image metadata used in the introductory notebooks."""
    if image is None:
        raise ValueError("Image is None. Check the image path.")

    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(image.dtype),
        "min_intensity": int(image.min()),
        "max_intensity": int(image.max()),
    }


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image

