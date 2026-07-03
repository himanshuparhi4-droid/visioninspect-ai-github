from ml.severity import calculate_severity, calculate_severity_from_prediction, score_from_defect_area_ratio


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
