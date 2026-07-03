from datetime import datetime

from beanie import Document
from pydantic import Field

from app.time_utils import utc_now


class Report(Document):
    inspection_id: str
    report_url: str
    report_path: str | None = None
    report_type: str = "pdf"
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "reports"
