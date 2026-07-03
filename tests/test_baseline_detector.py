import numpy as np

from ml.baseline_detector import anomaly_score, evaluate_binary_predictions, predict_from_score, threshold_from_scores


def test_baseline_threshold_and_prediction_helpers():
    scores = [1.0, 2.0, 3.0, 4.0]
    threshold = threshold_from_scores(scores, percentile=75)

    assert threshold == 3.25
    assert predict_from_score(4.0, threshold) == 1
    assert predict_from_score(2.0, threshold) == 0


def test_anomaly_score_uses_percentile():
    diff_map = np.array([[0, 10], [20, 30]], dtype=np.float32)

    assert anomaly_score(diff_map, percentile=50) == 15.0


def test_evaluate_binary_predictions():
    metrics = evaluate_binary_predictions([0, 0, 1, 1], [0, 1, 1, 1])

    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 1.0
