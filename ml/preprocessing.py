import cv2
import numpy as np


def resize_image(image: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    return cv2.resize(image, size)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_image(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 0)


def enhance_contrast(gray_image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def detect_edges(gray_image: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray_image, 100, 200)

