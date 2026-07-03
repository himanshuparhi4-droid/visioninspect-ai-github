from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_current_user, require_roles
from app.models.inspection_model import Inspection
from app.models.rework_model import ReworkTicket
from app.models.user_model import User
from app.schemas.rework_schema import ReworkTicketCreate, ReworkTicketResponse, ReworkTicketUpdate
from app.services.audit_service import record_audit_event
from app.services.rework_service import create_or_update_rework_ticket, rework_to_response
from app.time_utils import utc_now

router = APIRouter(prefix="/rework", tags=["rework"])

ADMIN_ROLES = {"admin", "quality_manager", "factory_supervisor"}


def parse_document_id(value: str) -> PydanticObjectId:
    try:
        return PydanticObjectId(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid document id") from exc


async def visible_ticket(ticket_id: str, current_user: User) -> ReworkTicket:
    ticket = await ReworkTicket.get(parse_document_id(ticket_id))
    if ticket is None:
        raise HTTPException(status_code=404, detail="Rework ticket not found")
    if current_user.role not in ADMIN_ROLES and ticket.created_by != str(current_user.id):
        raise HTTPException(status_code=403, detail="You cannot access this ticket")
    return ticket


@router.get("/tickets", response_model=list[ReworkTicketResponse])
async def list_rework_tickets(
    status: str | None = None,
    current_user: User = Depends(get_current_user),
) -> list[ReworkTicketResponse]:
    filters = []
    if current_user.role not in ADMIN_ROLES:
        filters.append(ReworkTicket.created_by == str(current_user.id))
    if status:
        filters.append(ReworkTicket.status == status)
    query = ReworkTicket.find(*filters) if filters else ReworkTicket.find_all()
    tickets = await query.sort("-created_at").to_list()
    return [ReworkTicketResponse(**rework_to_response(ticket)) for ticket in tickets]


@router.get("/tickets/by-inspection/{inspection_id}", response_model=ReworkTicketResponse)
async def get_rework_ticket_for_inspection(
    inspection_id: str,
    current_user: User = Depends(get_current_user),
) -> ReworkTicketResponse:
    inspection = await Inspection.get(parse_document_id(inspection_id))
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found")
    if current_user.role not in ADMIN_ROLES and inspection.uploaded_by != str(current_user.id):
        raise HTTPException(status_code=403, detail="You cannot access this inspection")

    ticket = await ReworkTicket.find(
        ReworkTicket.inspection_id == inspection_id,
        ReworkTicket.status != "closed",
    ).sort("-created_at").first_or_none()
    if ticket is None:
        raise HTTPException(status_code=404, detail="Rework ticket not found for this inspection")
    return ReworkTicketResponse(**rework_to_response(ticket))


@router.post("/tickets", response_model=ReworkTicketResponse, status_code=201)
async def create_rework_ticket(
    payload: ReworkTicketCreate,
    current_user: User = Depends(get_current_user),
) -> ReworkTicketResponse:
    inspection = await Inspection.get(parse_document_id(payload.inspection_id))
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found")
    if current_user.role not in ADMIN_ROLES and inspection.uploaded_by != str(current_user.id):
        raise HTTPException(status_code=403, detail="You cannot access this inspection")

    ticket = await create_or_update_rework_ticket(
        inspection=inspection,
        current_user=current_user,
        reason=payload.reason,
        assigned_to=payload.assigned_to,
    )
    if payload.priority:
        ticket.priority = payload.priority
    ticket.due_at = payload.due_at
    ticket.updated_at = utc_now()
    await ticket.save()
    inspection.review_status = "sent_for_rework"
    inspection.review_notes = payload.reason or inspection.review_notes or ticket.reason
    inspection.reviewed_by = str(current_user.id)
    inspection.reviewed_at = utc_now()
    inspection.updated_at = utc_now()
    await inspection.save()
    await record_audit_event(actor=current_user, action="rework.ticket_created", entity_type="rework_ticket", entity_id=str(ticket.id), metadata={"inspection_id": payload.inspection_id})
    return ReworkTicketResponse(**rework_to_response(ticket))


@router.patch("/tickets/{ticket_id}", response_model=ReworkTicketResponse)
async def update_rework_ticket(
    ticket_id: str,
    payload: ReworkTicketUpdate,
    current_user: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> ReworkTicketResponse:
    ticket = await visible_ticket(ticket_id, current_user)
    previous_status = ticket.status
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(ticket, field, value.strip() if isinstance(value, str) and value.strip() else value)
    if ticket.status == "in_progress" and ticket.started_at is None:
        ticket.started_at = utc_now()
    if ticket.status in {"completed", "closed"} and ticket.resolved_at is None:
        ticket.resolved_at = utc_now()
    ticket.updated_at = utc_now()
    await ticket.save()

    inspection = await Inspection.get(parse_document_id(ticket.inspection_id))
    if inspection is not None:
        if ticket.status in {"open", "in_progress"}:
            inspection.review_status = "sent_for_rework"
        elif ticket.status in {"completed", "closed"}:
            inspection.review_status = "manual_review"
            inspection.review_notes = ticket.resolution_notes or inspection.review_notes
        inspection.reviewed_by = str(current_user.id)
        inspection.reviewed_at = utc_now()
        inspection.updated_at = utc_now()
        await inspection.save()

    await record_audit_event(
        actor=current_user,
        action="rework.ticket_updated",
        entity_type="rework_ticket",
        entity_id=str(ticket.id),
        metadata={"status": ticket.status, "previous_status": previous_status, "inspection_id": ticket.inspection_id},
    )
    return ReworkTicketResponse(**rework_to_response(ticket))
