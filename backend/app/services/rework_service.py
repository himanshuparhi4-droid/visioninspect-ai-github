from uuid import uuid4

from app.models.inspection_model import Inspection
from app.models.rework_model import ReworkTicket
from app.models.user_model import User
from app.time_utils import utc_now


def priority_from_inspection(inspection: Inspection) -> str:
    if inspection.severity_level == "Critical":
        return "Critical"
    if inspection.severity_level == "High":
        return "High"
    if inspection.severity_level == "Medium":
        return "Medium"
    return "Low"


def rework_to_response(ticket: ReworkTicket) -> dict:
    return {
        "id": str(ticket.id),
        "ticket_number": ticket.ticket_number,
        "inspection_id": ticket.inspection_id,
        "created_by": ticket.created_by,
        "assigned_to": ticket.assigned_to,
        "product_id": ticket.product_id,
        "batch_number": ticket.batch_number,
        "production_line": ticket.production_line,
        "defect_type": ticket.defect_type,
        "severity_level": ticket.severity_level,
        "priority": ticket.priority,
        "status": ticket.status,
        "reason": ticket.reason,
        "resolution_notes": ticket.resolution_notes,
        "due_at": ticket.due_at,
        "started_at": ticket.started_at,
        "resolved_at": ticket.resolved_at,
        "created_at": ticket.created_at,
        "updated_at": ticket.updated_at,
    }


def generate_ticket_number() -> str:
    return f"RWK-{utc_now().strftime('%Y%m%d')}-{uuid4().hex[:6].upper()}"


async def create_or_update_rework_ticket(
    inspection: Inspection,
    current_user: User,
    reason: str | None = None,
    assigned_to: str | None = None,
) -> ReworkTicket:
    existing = await ReworkTicket.find(
        ReworkTicket.inspection_id == str(inspection.id),
        ReworkTicket.status != "completed",
        ReworkTicket.status != "closed",
    ).first_or_none()

    if existing:
        existing.ticket_number = existing.ticket_number or generate_ticket_number()
        existing.reason = reason or existing.reason
        existing.assigned_to = assigned_to or existing.assigned_to
        existing.priority = priority_from_inspection(inspection)
        existing.status = existing.status if existing.status in {"open", "in_progress"} else "open"
        existing.updated_at = utc_now()
        await existing.save()
        return existing

    ticket = ReworkTicket(
        ticket_number=generate_ticket_number(),
        inspection_id=str(inspection.id),
        created_by=str(current_user.id),
        assigned_to=assigned_to,
        product_id=inspection.product_id,
        batch_number=inspection.batch_number,
        production_line=inspection.production_line,
        defect_type=inspection.defect_type,
        severity_level=inspection.severity_level,
        priority=priority_from_inspection(inspection),
        status="open",
        reason=reason or inspection.review_notes or inspection.recommended_action,
    )
    await ticket.insert()
    return ticket
