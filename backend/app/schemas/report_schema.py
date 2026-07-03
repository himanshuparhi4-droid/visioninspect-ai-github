from datetime import datetime

from pydantic import BaseModel


class ReportResponse(BaseModel):
    id: str | None = None
    inspection_id: str
    report_url: str
    report_type: str = "pdf"
    created_at: datetime | None = None
