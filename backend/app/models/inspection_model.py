from datetime import datetime

from beanie import Document
from pydantic import Field

from app.time_utils import utc_now


class Inspection(Document):
    uploaded_by: str
    original_image_url: str
    original_image_path: str | None = None
    processed_image_url: str | None = None
    processed_image_path: str | None = None
    heatmap_url: str | None = None
    heatmap_path: str | None = None
    prediction: str | None = None
    defect_type: str | None = None
    confidence: float | None = None
    anomaly_score: float | None = None
    defect_area_ratio: float | None = None
    class_probabilities: dict = Field(default_factory=dict)
    severity_score: float | None = None
    severity_level: str | None = None
    severity_components: dict = Field(default_factory=dict)
    explainability: dict = Field(default_factory=dict)
    pass_fail: str | None = None
    recommended_action: str | None = None
    model_version: str | None = None
    review_status: str = "ai_pending"
    review_notes: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    batch_number: str | None = None
    product_id: str | None = None
    production_line: str | None = None
    shift: str | None = None
    operator_name: str | None = None
    source_type: str = "manual_upload"
    source_label: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "inspections"
