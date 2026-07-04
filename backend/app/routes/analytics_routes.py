import csv
from datetime import datetime
from io import StringIO

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.dependencies import get_current_user
from app.models.inspection_model import Inspection
from app.models.user_model import User
from app.services.analytics_service import build_analytics_summary

router = APIRouter(prefix="/analytics", tags=["analytics"])

ADMIN_ROLES = {"admin", "quality_manager", "factory_supervisor"}


@router.get("/summary")
async def analytics_summary(
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    production_line: str | None = None,
    product_id: str | None = None,
    current_user: User = Depends(get_current_user),
) -> dict:
    uploaded_by = None if current_user.role in ADMIN_ROLES else str(current_user.id)
    return await build_analytics_summary(
        uploaded_by=uploaded_by,
        date_from=date_from,
        date_to=date_to,
        production_line=production_line,
        product_id=product_id,
    )


@router.get("/export.csv")
async def export_inspections_csv(current_user: User = Depends(get_current_user)) -> StreamingResponse:
    if current_user.role in ADMIN_ROLES:
        inspections = await Inspection.find_all().sort("-created_at").to_list()
    else:
        inspections = (
            await Inspection.find(Inspection.uploaded_by == str(current_user.id)).sort("-created_at").to_list()
        )

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "inspection_id",
            "created_at",
            "product_id",
            "batch_number",
            "production_line",
            "shift",
            "operator_name",
            "source_type",
            "prediction",
            "defect_type",
            "confidence",
            "severity_score",
            "severity_level",
            "pass_fail",
            "review_status",
        ]
    )
    for item in inspections:
        writer.writerow(
            [
                str(item.id),
                item.created_at.isoformat(),
                item.product_id or "",
                item.batch_number or "",
                item.production_line or "",
                item.shift or "",
                item.operator_name or "",
                item.source_type or "",
                item.prediction or "",
                item.defect_type or "",
                item.confidence or "",
                item.severity_score or "",
                item.severity_level or "",
                item.pass_fail or "",
                item.review_status or "",
            ]
        )

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=visioninspect_inspections.csv"},
    )
