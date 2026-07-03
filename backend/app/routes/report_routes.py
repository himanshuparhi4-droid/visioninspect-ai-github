from pathlib import Path

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.dependencies import get_current_user
from app.models.inspection_model import Inspection
from app.models.report_model import Report
from app.models.user_model import User
from app.schemas.report_schema import ReportResponse
from app.serializers import report_to_response
from app.services.audit_service import record_audit_event
from app.services.report_service import generate_inspection_report_pdf, inspection_report_path

router = APIRouter(prefix="/reports", tags=["reports"])

ADMIN_ROLES = {"admin", "quality_manager", "factory_supervisor"}


def parse_document_id(value: str) -> PydanticObjectId:
    try:
        return PydanticObjectId(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid document id") from exc


async def get_allowed_inspection(inspection_id: str, current_user: User) -> Inspection:
    inspection = await Inspection.get(parse_document_id(inspection_id))
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found")
    if current_user.role not in ADMIN_ROLES and inspection.uploaded_by != str(current_user.id):
        raise HTTPException(status_code=403, detail="You cannot access this inspection")
    return inspection


@router.post("/inspection/{inspection_id}", response_model=ReportResponse, status_code=201)
async def create_report(
    inspection_id: str,
    current_user: User = Depends(get_current_user),
) -> ReportResponse:
    inspection = await get_allowed_inspection(inspection_id, current_user)
    report_path = generate_inspection_report_pdf(inspection)
    report = Report(
        inspection_id=str(inspection.id),
        report_url="",
        report_path=str(report_path),
        report_type="pdf",
    )
    await report.insert()
    report.report_url = f"{settings.backend_url.rstrip('/')}/reports/{report.id}/download"
    await report.save()
    await record_audit_event(
        actor=current_user,
        action="report.generated",
        entity_type="report",
        entity_id=str(report.id),
        metadata={"inspection_id": str(inspection.id), "report_type": report.report_type},
    )
    return report_to_response(report)


@router.get("/{report_id}/download")
async def download_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    report = await Report.get(parse_document_id(report_id))
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    inspection = await get_allowed_inspection(report.inspection_id, current_user)
    report_path = Path(report.report_path) if report.report_path else inspection_report_path(inspection)
    if not report_path.exists():
        report_path = generate_inspection_report_pdf(inspection)
        report.report_path = str(report_path)
        await report.save()

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=f"visioninspect_report_{inspection.id}.pdf",
    )


@router.get("", response_model=list[ReportResponse])
async def list_reports(current_user: User = Depends(get_current_user)) -> list[ReportResponse]:
    if current_user.role in ADMIN_ROLES:
        reports = await Report.find_all().sort("-created_at").to_list()
    else:
        inspections = await Inspection.find(Inspection.uploaded_by == str(current_user.id)).to_list()
        inspection_ids = {str(inspection.id) for inspection in inspections}
        reports = await Report.find(Report.inspection_id.in_(inspection_ids)).sort("-created_at").to_list()
    return [report_to_response(report) for report in reports]


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
) -> ReportResponse:
    report = await Report.get(parse_document_id(report_id))
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    await get_allowed_inspection(report.inspection_id, current_user)
    return report_to_response(report)
