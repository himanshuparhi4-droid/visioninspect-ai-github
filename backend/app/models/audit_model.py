from datetime import datetime

from beanie import Document
from pydantic import Field

from app.time_utils import utc_now


class AuditLog(Document):
    actor_id: str | None = None
    action: str
    entity_type: str
    entity_id: str | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "audit_logs"
