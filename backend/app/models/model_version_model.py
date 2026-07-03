from datetime import datetime

from beanie import Document
from pydantic import Field

from app.time_utils import utc_now


class ModelVersion(Document):
    name: str
    version: str
    artifact_path: str
    metrics: dict = Field(default_factory=dict)
    trained_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "model_versions"
