from pathlib import Path

import numpy as np

from ml.baseline_detector import (
    anomaly_mask,
    anomaly_score,
    evaluate_binary_predictions,
    predict_from_score,
    threshold_from_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_baseline_threshold_and_prediction_helpers():
    scores = [1.0, 2.0, 3.0, 4.0]
    threshold = threshold_from_scores(scores, percentile=75)

    assert threshold == 3.25
    assert predict_from_score(4.0, threshold) == 1
    assert predict_from_score(2.0, threshold) == 0


def test_anomaly_score_uses_percentile():
    diff_map = np.array([[0, 10], [20, 30]], dtype=np.float32)

    assert anomaly_score(diff_map, percentile=50) == 15.0
    assert anomaly_score(diff_map, percentile=50, mask=np.array([[True, False], [False, True]])) == 15.0


def test_anomaly_mask_filters_tiny_residual_noise():
    diff_map = np.zeros((8, 8), dtype=np.float32)
    diff_map[0, 0] = 9.0
    diff_map[3:6, 3:6] = 9.0

    mask = anomaly_mask(diff_map, score_threshold=1.45)

    assert not mask[0, 0]
    assert int(mask.sum()) == 9


def test_saved_normal_profile_has_runtime_arrays():
    from ml.baseline_detector import load_reference_profile

    profile = load_reference_profile(PROJECT_ROOT / "models" / "inference" / "normal_profile.npz")

    assert profile["mean"].shape == (256, 256)
    assert profile["std"].shape == (256, 256)
    assert profile["foreground_mask"].shape == (256, 256)
    assert profile["foreground_mask"].any()


def test_evaluate_binary_predictions():
    metrics = evaluate_binary_predictions([0, 0, 1, 1], [0, 1, 1, 1])

    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 1.0
