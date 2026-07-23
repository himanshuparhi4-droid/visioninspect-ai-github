from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

DEFAULT_VARIABILITY_FLOOR = 15.0


def load_image_bgr(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def preprocess_gray(image_bgr: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    resized = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced.astype(np.float32)


def build_reference_image(train_image_paths: list[str | Path], size: tuple[int, int] = (256, 256)) -> np.ndarray:
    return build_reference_profile(train_image_paths, size=size)["mean"]


def foreground_mask(reference_image: np.ndarray) -> np.ndarray:
    """Find stable product pixels and exclude the bright background."""
    reference_uint8 = np.clip(reference_image, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(reference_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask.astype(bool)


def build_reference_profile(train_image_paths: list[str | Path], size: tuple[int, int] = (256, 256)) -> dict:
    """Create a normal-image profile used by the lightweight fallback detector."""
    if not train_image_paths:
        raise ValueError("No training images provided.")

    processed_images = [preprocess_gray(load_image_bgr(image_path), size=size) for image_path in train_image_paths]
    image_stack = np.stack(processed_images).astype(np.float32)
    mean_image = np.mean(image_stack, axis=0).astype(np.float32)

    return {
        "mean": mean_image,
        "std": np.std(image_stack, axis=0).astype(np.float32),
        "foreground_mask": foreground_mask(mean_image),
    }


def save_reference_profile(profile: dict, destination: str | Path) -> None:
    """Persist only the compact normal-profile arrays required at runtime."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=np.asarray(profile["mean"], dtype=np.float32),
        std=np.asarray(profile["std"], dtype=np.float32),
        foreground_mask=np.asarray(profile["foreground_mask"], dtype=bool),
    )


def load_reference_profile(profile_path: str | Path) -> dict:
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline profile not found: {path}")

    with np.load(path) as profile:
        return {
            "mean": profile["mean"].astype(np.float32),
            "std": profile["std"].astype(np.float32),
            "foreground_mask": profile["foreground_mask"].astype(bool),
        }


def anomaly_map(image_bgr: np.ndarray, reference_image: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    processed = preprocess_gray(image_bgr, size=size)
    diff = cv2.absdiff(processed, reference_image)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    return diff


def normalized_anomaly_map(
    image_bgr: np.ndarray,
    reference_profile: dict,
    size: tuple[int, int] = (256, 256),
    variability_floor: float = DEFAULT_VARIABILITY_FLOOR,
) -> np.ndarray:
    """Score residuals relative to normal variation after lighting correction."""
    processed = preprocess_gray(image_bgr, size=size)
    reference = np.asarray(reference_profile["mean"], dtype=np.float32)
    variability = np.asarray(reference_profile["std"], dtype=np.float32)
    mask = np.asarray(reference_profile["foreground_mask"], dtype=bool)

    if processed.shape != reference.shape or reference.shape != variability.shape or mask.shape != reference.shape:
        raise ValueError("Baseline profile dimensions do not match the configured image size.")
    if not np.any(mask):
        raise ValueError("Baseline profile foreground mask is empty.")

    brightness_offset = float(np.median((processed - reference)[mask]))
    residual = np.abs(processed - brightness_offset - reference) / (variability + variability_floor)
    residual = cv2.GaussianBlur(residual.astype(np.float32), (5, 5), 0)
    residual[~mask] = 0.0
    return residual.astype(np.float32)


def anomaly_score(diff_map: np.ndarray, percentile: float = 99.0, mask: np.ndarray | None = None) -> float:
    values = diff_map[np.asarray(mask, dtype=bool)] if mask is not None else diff_map.reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def anomaly_mask(diff_map: np.ndarray, score_threshold: float) -> np.ndarray:
    """Remove low-level residual noise before geometry and heatmap reporting."""
    pixel_threshold = max(2.5, float(score_threshold) * 1.75)
    candidate = (diff_map > pixel_threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    cleaned = np.zeros_like(candidate, dtype=bool)
    for label_index in range(1, count):
        if stats[label_index, cv2.CC_STAT_AREA] >= 4:
            cleaned[labels == label_index] = True
    return cleaned


def threshold_from_scores(scores: list[float], percentile: float = 99.0) -> float:
    if not scores:
        raise ValueError("No scores provided for threshold calculation.")
    return float(np.percentile(scores, percentile))


def predict_from_score(score: float, threshold: float) -> int:
    return int(score > threshold)


def evaluate_binary_predictions(y_true: list[int], y_pred: list[int]) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": matrix.tolist(),
    }


def heatmap_overlay(image_bgr: np.ndarray, diff_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_resized = cv2.resize(image_bgr, (diff_map.shape[1], diff_map.shape[0]), interpolation=cv2.INTER_AREA)
    normalized = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_resized, 1 - alpha, heatmap, alpha, 0)
    return overlay


def score_dataframe(dataset_df: pd.DataFrame, reference_image: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = []
    for _, row in dataset_df.iterrows():
        image_bgr = load_image_bgr(row["image_path"])
        diff = anomaly_map(image_bgr, reference_image)
        score = anomaly_score(diff)
        prediction = predict_from_score(score, threshold)
        rows.append(
            {
                **row.to_dict(),
                "anomaly_score": score,
                "prediction": prediction,
                "prediction_name": "defective" if prediction == 1 else "good",
            }
        )
    return pd.DataFrame(rows)
