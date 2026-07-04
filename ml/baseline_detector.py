from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


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
    if not train_image_paths:
        raise ValueError("No training images provided.")

    processed_images = []
    for image_path in train_image_paths:
        image_bgr = load_image_bgr(image_path)
        processed_images.append(preprocess_gray(image_bgr, size=size))

    return np.mean(processed_images, axis=0).astype(np.float32)


def anomaly_map(image_bgr: np.ndarray, reference_image: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    processed = preprocess_gray(image_bgr, size=size)
    diff = cv2.absdiff(processed, reference_image)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    return diff


def anomaly_score(diff_map: np.ndarray, percentile: float = 99.0) -> float:
    return float(np.percentile(diff_map, percentile))


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
