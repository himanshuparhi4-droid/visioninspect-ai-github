from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ReviewStatus = Literal[
    "uploaded",
    "ai_pending",
    "ai_completed",
    "manual_review",
    "approved",
    "rejected",
    "sent_for_rework",
    "re_inspected",
]
ReviewAction = Literal["approved", "rejected", "sent_for_rework"]


class InspectionResponse(BaseModel):
    id: str
    uploaded_by: str
    original_image_url: str
    processed_image_url: str | None = None
    heatmap_url: str | None = None
    prediction: str | None = None
    defect_type: str | None = None
    candidate_defect_type: str | None = None
    confidence: float | None = None
    detection_confidence: float | None = None
    classification_confidence: float | None = None
    raw_classification_confidence: float | None = None
    classification_confidence_calibrated: bool = False
    subtype_model_status: str | None = None
    subtype_model_macro_f1: float | None = None
    manual_review_required: bool = False
    detector_engine: str | None = None
    detector_kind: str | None = None
    classifier_engine: str | None = None
    detector_fallback_used: bool = False
    detector_fallback_reason: str | None = None
    classifier_fallback_used: bool = False
    classifier_fallback_reason: str | None = None
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
    review_status: str
    review_notes: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    rework_ticket_id: str | None = None
    rework_ticket_number: str | None = None
    rework_ticket_status: str | None = None
    batch_number: str | None = None
    product_id: str | None = None
    production_line: str | None = None
    shift: str | None = None
    operator_name: str | None = None
    source_type: str
    source_label: str | None = None
    category: str | None = None
    created_at: datetime
    updated_at: datetime


class InspectionListResponse(BaseModel):
    total: int
    items: list[InspectionResponse]
    summary: dict = Field(default_factory=dict)
    failures: list[dict] = Field(default_factory=list)


class ReviewStatusUpdate(BaseModel):
    review_status: ReviewAction
    review_notes: str | None = Field(default=None, max_length=2000)


class InspectionMetadataUpdate(BaseModel):
    batch_number: str | None = Field(default=None, max_length=120)
    product_id: str | None = Field(default=None, max_length=120)
    production_line: str | None = Field(default=None, max_length=120)
    shift: str | None = Field(default=None, max_length=120)
    operator_name: str | None = Field(default=None, max_length=160)
    source_label: str | None = Field(default=None, max_length=240)
    category: str | None = Field(default=None, max_length=80)
