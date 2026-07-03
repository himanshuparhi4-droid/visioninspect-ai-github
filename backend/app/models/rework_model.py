from datetime import datetime

from beanie import Document
from pydantic import Field

from app.time_utils import utc_now


class ReworkTicket(Document):
    ticket_number: str | None = None
    inspection_id: str
    created_by: str
    assigned_to: str | None = None
    product_id: str | None = None
    batch_number: str | None = None
    production_line: str | None = None
    defect_type: str | None = None
    severity_level: str | None = None
    priority: str = "Medium"
    status: str = "open"
    reason: str | None = None
    resolution_notes: str | None = None
    due_at: datetime | None = None
    started_at: datetime | None = None
    resolved_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "rework_tickets"
