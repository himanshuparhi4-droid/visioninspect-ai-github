import numpy as np

from ml.heatmap import colorize_anomaly_map, localization_metrics, mask_overlay, normalize_anomaly_map, overlay_heatmap


def test_heatmap_utilities_return_visualization_shapes():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :] = [20, 80, 140]
    anomaly_map = np.linspace(0, 1, 16 * 24, dtype=np.float32).reshape(16, 24)

    normalized = normalize_anomaly_map(anomaly_map)
    heatmap = colorize_anomaly_map(anomaly_map)
    overlay = overlay_heatmap(image, anomaly_map)

    assert normalized.shape == anomaly_map.shape
    assert normalized.dtype == np.uint8
    assert heatmap.shape == (16, 24, 3)
    assert overlay.shape == (16, 24, 3)
    assert overlay.dtype == np.uint8


def test_mask_overlay_and_localization_metrics():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt[2:6, 2:6] = 1
    pred[4:8, 4:8] = 1

    overlay = mask_overlay(image, gt, color=(255, 0, 0))
    metrics = localization_metrics(gt, pred)

    assert overlay.shape == image.shape
    assert metrics["iou"] == 0.1429
    assert metrics["dice"] == 0.25
    assert metrics["ground_truth_coverage"] == 0.25
