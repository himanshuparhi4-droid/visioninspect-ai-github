from __future__ import annotations

import cv2
import numpy as np


def normalize_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
    """Scale an anomaly map to uint8 range for visualization."""
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    if anomaly_map.size == 0:
        raise ValueError("anomaly_map cannot be empty.")

    minimum = float(np.min(anomaly_map))
    maximum = float(np.max(anomaly_map))
    if maximum == minimum:
        return np.zeros(anomaly_map.shape, dtype=np.uint8)

    normalized = (anomaly_map - minimum) / (maximum - minimum)
    return np.clip(normalized * 255, 0, 255).astype(np.uint8)


def colorize_anomaly_map(anomaly_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a single-channel anomaly map into an RGB heatmap."""
    normalized = normalize_anomaly_map(anomaly_map)
    heatmap_bgr = cv2.applyColorMap(normalized, colormap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(image_rgb: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay an RGB heatmap on an RGB image."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an RGB image with shape HxWx3.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    heatmap_rgb = colorize_anomaly_map(anomaly_map)
    image_resized = cv2.resize(image_rgb, (heatmap_rgb.shape[1], heatmap_rgb.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(image_resized, 1 - alpha, heatmap_rgb, alpha, 0)


def mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary mask on an RGB image."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an RGB image with shape HxWx3.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.shape != image_rgb.shape[:2]:
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay = image_rgb.copy()
    color_layer = np.zeros_like(overlay)
    color_layer[:, :] = np.array(color, dtype=np.uint8)
    return np.where(mask_bool[..., None], (overlay * (1 - alpha) + color_layer * alpha).astype(np.uint8), overlay)


def localization_metrics(ground_truth_mask: np.ndarray, predicted_mask: np.ndarray) -> dict:
    """Calculate simple mask overlap metrics for localization evidence."""
    gt = np.asarray(ground_truth_mask).astype(bool)
    pred = np.asarray(predicted_mask).astype(bool)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    intersection = int(np.logical_and(gt, pred).sum())
    union = int(np.logical_or(gt, pred).sum())
    gt_area = int(gt.sum())
    pred_area = int(pred.sum())

    return {
        "iou": round(float(intersection / union), 4) if union else 0.0,
        "dice": round(float((2 * intersection) / (gt_area + pred_area)), 4) if (gt_area + pred_area) else 0.0,
        "ground_truth_coverage": round(float(intersection / gt_area), 4) if gt_area else 0.0,
        "ground_truth_area_ratio": round(float(gt.mean()), 4),
        "predicted_area_ratio": round(float(pred.mean()), 4),
    }
