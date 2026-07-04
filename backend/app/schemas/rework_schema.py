from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ReworkStatus = Literal["open", "in_progress", "completed", "closed"]
ReworkPriority = Literal["Low", "Medium", "High", "Critical"]


class ReworkTicketCreate(BaseModel):
    inspection_id: str
    assigned_to: str | None = Field(default=None, max_length=160)
    priority: ReworkPriority | None = None
    reason: str | None = Field(default=None, max_length=2000)
    due_at: datetime | None = None


class ReworkTicketUpdate(BaseModel):
    assigned_to: str | None = Field(default=None, max_length=160)
    priority: ReworkPriority | None = None
    status: ReworkStatus | None = None
    reason: str | None = Field(default=None, max_length=2000)
    resolution_notes: str | None = Field(default=None, max_length=2000)
    due_at: datetime | None = None


class ReworkTicketResponse(BaseModel):
    id: str
    ticket_number: str | None = None
    inspection_id: str
    created_by: str
    assigned_to: str | None = None
    product_id: str | None = None
    batch_number: str | None = None
    production_line: str | None = None
    defect_type: str | None = None
    severity_level: str | None = None
    priority: str
    status: str
    reason: str | None = None
    resolution_notes: str | None = None
    due_at: datetime | None = None
    started_at: datetime | None = None
    resolved_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
