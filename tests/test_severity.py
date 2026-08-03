from pathlib import Path

import joblib

from ml.severity import (
    DEFECT_TYPE_SCORES,
    calculate_severity,
    calculate_severity_from_prediction,
    score_from_defect_area_ratio,
)


def test_calculate_severity_critical():
    result = calculate_severity(85, 90, 95, 92)
    assert result["severity_level"] == "Critical"
    assert result["pass_fail"] == "Fail"


def test_area_ratio_size_score():
    assert score_from_defect_area_ratio(0) == 0
    assert score_from_defect_area_ratio(0.01) == 45
    assert score_from_defect_area_ratio(0.08) == 90


def test_prediction_to_severity():
    result = calculate_severity_from_prediction(
        defect_type="broken_large",
        confidence=0.96,
        area_ratio=0.085,
        is_critical_location=True,
    )

    assert result["severity_level"] == "Critical"
    assert result["pass_fail"] == "Fail"
    assert result["components"]["defect_type_score"] == 95


def test_good_prediction_has_zero_defect_severity():
    result = calculate_severity_from_prediction(
        defect_type="good",
        confidence=0.73,
        area_ratio=0.0,
        is_critical_location=False,
    )

    assert result["severity_score"] == 0.0
    assert result["pass_fail"] == "Pass"
    assert all(value == 0 for value in result["components"].values())


def test_classifier_labels_have_explicit_severity_scores():
    labels = set()
    for path in [Path("models/defect_classifier.pkl"), *Path("models/categories").glob("*/defect_classifier.pkl")]:
        if not path.exists():
            continue
        artifact = joblib.load(path)
        classifier = artifact.get("classifier") if isinstance(artifact, dict) else artifact
        if hasattr(classifier, "classes_"):
            labels.update(str(label) for label in classifier.classes_)

    assert labels
    assert labels - set(DEFECT_TYPE_SCORES) == set()
