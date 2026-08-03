from collections import Counter
from datetime import datetime

from app.models.inspection_model import Inspection
from app.time_utils import as_utc


async def build_analytics_summary(
    uploaded_by: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    production_line: str | None = None,
    product_id: str | None = None,
    defect_type: str | None = None,
) -> dict:
    if uploaded_by:
        query = Inspection.find(Inspection.uploaded_by == uploaded_by)
    else:
        query = Inspection.find_all()

    inspections = await query.to_list()
    if date_from:
        normalized_from = as_utc(date_from)
        inspections = [item for item in inspections if as_utc(item.created_at) >= normalized_from]
    if date_to:
        normalized_to = as_utc(date_to)
        inspections = [item for item in inspections if as_utc(item.created_at) <= normalized_to]
    if production_line:
        inspections = [item for item in inspections if item.production_line == production_line]
    if product_id:
        inspections = [item for item in inspections if item.product_id == product_id]
    if defect_type:
        inspections = [item for item in inspections if item.defect_type == defect_type]

    total = len(inspections)
    defective_count = sum(1 for item in inspections if item.prediction == "Defective")
    good_count = sum(1 for item in inspections if item.prediction == "Good")
    review_count = sum(1 for item in inspections if item.pass_fail == "Review")
    fail_count = sum(1 for item in inspections if item.pass_fail == "Fail")
    pass_count = sum(1 for item in inspections if item.pass_fail == "Pass")
    detection_confidences = [
        item.detection_confidence if item.detection_confidence is not None else item.confidence
        for item in inspections
        if item.detection_confidence is not None or item.confidence is not None
    ]
    classification_confidences = [
        item.classification_confidence
        for item in inspections
        if item.classification_confidence is not None
    ]
    severity_scores = [item.severity_score for item in inspections if item.severity_score is not None]

    defect_types = Counter(item.defect_type or "unknown" for item in inspections)
    severity_levels = Counter(item.severity_level or "pending" for item in inspections)
    review_statuses = Counter(item.review_status or "pending" for item in inspections)
    production_lines = Counter(item.production_line or "unassigned" for item in inspections)
    source_types = Counter(item.source_type or "unknown" for item in inspections)
    shifts = Counter(item.shift or "unassigned" for item in inspections)

    trend_map: dict[str, dict] = {}
    defect_type_by_line: dict[str, Counter] = {}
    severity_by_line: dict[str, Counter] = {}
    line_stats: dict[str, dict] = {}
    for item in inspections:
        day = as_utc(item.created_at).strftime("%Y-%m-%d")
        row = trend_map.setdefault(
            day,
            {
                "date": day,
                "total": 0,
                "defective": 0,
                "good": 0,
                "pass": 0,
                "review": 0,
                "fail": 0,
                "defect_types": Counter(),
                "severity_levels": Counter(),
            },
        )
        row["total"] += 1
        if item.prediction == "Defective":
            row["defective"] += 1
        if item.prediction == "Good":
            row["good"] += 1
        if item.pass_fail == "Pass":
            row["pass"] += 1
        if item.pass_fail == "Review":
            row["review"] += 1
        if item.pass_fail == "Fail":
            row["fail"] += 1
        row["defect_types"][item.defect_type or "unknown"] += 1
        row["severity_levels"][item.severity_level or "pending"] += 1

        line = item.production_line or "unassigned"
        defect_type_by_line.setdefault(line, Counter())[item.defect_type or "unknown"] += 1
        severity_by_line.setdefault(line, Counter())[item.severity_level or "pending"] += 1

        stats = line_stats.setdefault(
            line,
            {
                "line": line,
                "total": 0,
                "pass": 0,
                "review": 0,
                "fail": 0,
                "defective": 0,
                "good": 0,
                "severity_scores": [],
            },
        )
        stats["total"] += 1
        if item.pass_fail == "Pass":
            stats["pass"] += 1
        if item.pass_fail == "Review":
            stats["review"] += 1
        if item.pass_fail == "Fail":
            stats["fail"] += 1
        if item.prediction == "Defective":
            stats["defective"] += 1
        if item.prediction == "Good":
            stats["good"] += 1
        if item.severity_score is not None:
            stats["severity_scores"].append(item.severity_score)

    trend_rows = []
    for key in sorted(trend_map):
        row = trend_map[key]
        row["defect_rate"] = round(row["defective"] / row["total"], 4) if row["total"] else 0.0
        row["pass_rate"] = round(row["pass"] / row["total"], 4) if row["total"] else 0.0
        row["fail_rate"] = round(row["fail"] / row["total"], 4) if row["total"] else 0.0
        row["review_rate"] = round(row["review"] / row["total"], 4) if row["total"] else 0.0
        row["defect_types"] = dict(row["defect_types"])
        row["severity_levels"] = dict(row["severity_levels"])
        trend_rows.append(row)

    line_performance = []
    for _line, stats in sorted(line_stats.items()):
        line_total = stats["total"]
        severity_values = stats.pop("severity_scores")
        line_performance.append(
            {
                **stats,
                "pass_rate": round(stats["pass"] / line_total, 4) if line_total else 0.0,
                "review_rate": round(stats["review"] / line_total, 4) if line_total else 0.0,
                "fail_rate": round(stats["fail"] / line_total, 4) if line_total else 0.0,
                "defect_rate": round(stats["defective"] / line_total, 4) if line_total else 0.0,
                "average_severity_score": round(sum(severity_values) / len(severity_values), 2)
                if severity_values
                else 0.0,
            }
        )

    return {
        "total_inspections": total,
        "defective_count": defective_count,
        "good_count": good_count,
        "pass_count": pass_count,
        "review_count": review_count,
        "fail_count": fail_count,
        "defect_rate": round(defective_count / total, 4) if total else 0.0,
        "average_confidence": (
            round(sum(detection_confidences) / len(detection_confidences), 4) if detection_confidences else 0.0
        ),
        "average_detection_confidence": (
            round(sum(detection_confidences) / len(detection_confidences), 4) if detection_confidences else 0.0
        ),
        "average_classification_confidence": (
            round(sum(classification_confidences) / len(classification_confidences), 4)
            if classification_confidences
            else 0.0
        ),
        "average_severity_score": round(sum(severity_scores) / len(severity_scores), 2) if severity_scores else 0.0,
        "defect_type_distribution": dict(defect_types),
        "severity_distribution": dict(severity_levels),
        "review_status_distribution": dict(review_statuses),
        "production_line_distribution": dict(production_lines),
        "source_type_distribution": dict(source_types),
        "shift_distribution": dict(shifts),
        "critical_count": sum(1 for item in inspections if item.severity_level == "Critical"),
        "manual_review_queue": sum(1 for item in inspections if item.review_status == "manual_review"),
        "rework_queue": sum(1 for item in inspections if item.review_status == "sent_for_rework"),
        "trend_by_day": trend_rows,
        "defect_type_by_line": {line: dict(counter) for line, counter in defect_type_by_line.items()},
        "severity_by_line": {line: dict(counter) for line, counter in severity_by_line.items()},
        "line_performance": line_performance,
    }
