from __future__ import annotations

DEFECT_TYPE_SCORES = {
    "good": 0,
    "contamination": 55,
    "defective": 70,
    "unknown_defect": 60,
    "broken_small": 75,
    "broken_large": 95,
    "broken": 90,
    "broken_teeth": 90,
    "crack": 95,
    "surface_crack": 95,
    "scratch": 35,
    "scratch_head": 45,
    "scratch_neck": 40,
    "missing_component": 95,
    "missing_cable": 95,
    "missing_wire": 95,
    "bent_wire": 85,
    "cable_swap": 90,
    "cut_inner_insulation": 95,
    "cut_outer_insulation": 90,
    "cut_lead": 90,
    "poke_insulation": 75,
    "poke": 75,
    "faulty_imprint": 60,
    "squeeze": 65,
    "color": 45,
    "cut": 85,
    "hole": 90,
    "metal_contamination": 80,
    "thread": 40,
    "thread_side": 40,
    "thread_top": 45,
    "print": 50,
    "bent": 80,
    "bent_lead": 85,
    "damaged_case": 90,
    "misplaced": 85,
    "flip": 75,
    "pill_type": 95,
    "combined": 95,
    "liquid": 85,
    "oil": 65,
    "glue": 45,
    "glue_strip": 45,
    "fold": 55,
    "manipulated_front": 80,
    "fabric_border": 50,
    "fabric_interior": 60,
    "rough": 45,
    "gray_stroke": 40,
    "color_triple": 70,
    "split_teeth": 90,
    "squeezed_teeth": 85,
}


def score_from_defect_area_ratio(area_ratio: float) -> float:
    """Convert defect area ratio into a 0-100 size score."""
    if area_ratio <= 0:
        return 0
    if area_ratio < 0.005:
        return 20
    if area_ratio < 0.02:
        return 45
    if area_ratio < 0.05:
        return 70
    return 90


def score_from_location(is_critical_location: bool) -> float:
    """Score location risk from product-specific critical-zone overlap."""
    if is_critical_location:
        return 90
    return 35


def score_from_defect_type(defect_type: str) -> float:
    return float(DEFECT_TYPE_SCORES.get(defect_type.lower(), 60))


def score_from_confidence(confidence: float) -> float:
    """Convert confidence in 0-1 or 0-100 format into a 0-100 score."""
    confidence_score = confidence * 100 if confidence <= 1 else confidence
    return float(max(0, min(100, confidence_score)))


def severity_level_from_score(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def action_from_severity(level: str) -> str:
    actions = {
        "Critical": "Reject product and trigger quality inspection workflow",
        "High": "Repair or rework recommended",
        "Medium": "Inspection review required",
        "Low": "Product generally acceptable",
    }
    return actions[level]


def pass_fail_from_severity(level: str) -> str:
    if level in {"Critical", "High"}:
        return "Fail"
    if level == "Medium":
        return "Review"
    return "Pass"


def calculate_severity(
    size_score: float,
    location_score: float,
    defect_type_score: float,
    confidence_score: float,
) -> dict:
    score = size_score * 0.30 + location_score * 0.25 + defect_type_score * 0.25 + confidence_score * 0.20
    level = severity_level_from_score(score)

    return {
        "severity_score": round(score, 2),
        "severity_level": level,
        "pass_fail": pass_fail_from_severity(level),
        "recommended_action": action_from_severity(level),
        "components": {
            "size_score": round(float(size_score), 2),
            "location_score": round(float(location_score), 2),
            "defect_type_score": round(float(defect_type_score), 2),
            "confidence_score": round(float(confidence_score), 2),
        },
    }


def calculate_severity_from_prediction(
    defect_type: str,
    confidence: float,
    area_ratio: float = 0.0,
    is_critical_location: bool = False,
    defect_center_y_ratio: float | None = None,
) -> dict:
    if defect_type.lower() == "good":
        return {
            "severity_score": 0.0,
            "severity_level": "Low",
            "pass_fail": "Pass",
            "recommended_action": "Product generally acceptable",
            "components": {
                "size_score": 0.0,
                "location_score": 0.0,
                "defect_type_score": 0.0,
                "confidence_score": 0.0,
            },
            "defect_type": "good",
            "confidence": round(float(confidence), 4),
            "area_ratio": 0.0,
            "is_critical_location": False,
            "defect_center_y_ratio": None,
        }

    size_score = score_from_defect_area_ratio(area_ratio)
    location_score = score_from_location(is_critical_location)
    defect_type_score = score_from_defect_type(defect_type)
    confidence_score = score_from_confidence(confidence)

    result = calculate_severity(
        size_score=size_score,
        location_score=location_score,
        defect_type_score=defect_type_score,
        confidence_score=confidence_score,
    )
    result.update(
        {
            "defect_type": defect_type,
            "confidence": round(float(confidence), 4),
            "area_ratio": round(float(area_ratio), 4),
            "is_critical_location": bool(is_critical_location),
            "defect_center_y_ratio": None if defect_center_y_ratio is None else round(float(defect_center_y_ratio), 4),
        }
    )
    return result
