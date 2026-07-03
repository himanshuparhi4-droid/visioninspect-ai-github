from app.models.inspection_model import Inspection
from app.models.report_model import Report
from app.models.user_model import User
from app.schemas.inspection_schema import InspectionResponse
from app.schemas.report_schema import ReportResponse
from app.schemas.user_schema import UserResponse


def user_to_response(user: User) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        name=user.name,
        email=user.email,
        role=user.role,
        requested_role=user.requested_role,
        approval_status=user.approval_status,
        approved_by=user.approved_by,
        approved_at=user.approved_at,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login_at=user.last_login_at,
    )


def inspection_to_response(inspection: Inspection) -> InspectionResponse:
    return InspectionResponse(
        id=str(inspection.id),
        uploaded_by=inspection.uploaded_by,
        original_image_url=inspection.original_image_url,
        processed_image_url=inspection.processed_image_url,
        heatmap_url=inspection.heatmap_url,
        prediction=inspection.prediction,
        defect_type=inspection.defect_type,
        confidence=inspection.confidence,
        anomaly_score=inspection.anomaly_score,
        defect_area_ratio=inspection.defect_area_ratio,
        class_probabilities=inspection.class_probabilities,
        severity_score=inspection.severity_score,
        severity_level=inspection.severity_level,
        severity_components=inspection.severity_components,
        explainability=inspection.explainability,
        pass_fail=inspection.pass_fail,
        recommended_action=inspection.recommended_action,
        model_version=inspection.model_version,
        review_status=inspection.review_status,
        review_notes=inspection.review_notes,
        reviewed_by=inspection.reviewed_by,
        reviewed_at=inspection.reviewed_at,
        rework_ticket_id=None,
        rework_ticket_number=None,
        rework_ticket_status=None,
        batch_number=inspection.batch_number,
        product_id=inspection.product_id,
        production_line=inspection.production_line,
        shift=inspection.shift,
        operator_name=inspection.operator_name,
        source_type=inspection.source_type,
        source_label=inspection.source_label,
        created_at=inspection.created_at,
        updated_at=inspection.updated_at,
    )


def report_to_response(report: Report) -> ReportResponse:
    return ReportResponse(
        id=str(report.id),
        inspection_id=report.inspection_id,
        report_url=report.report_url,
        report_type=report.report_type,
        created_at=report.created_at,
    )
