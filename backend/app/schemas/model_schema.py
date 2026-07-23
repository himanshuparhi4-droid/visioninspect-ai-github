from pydantic import BaseModel, Field


class RuntimeModelSettings(BaseModel):
    padim_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    baseline_threshold: float = Field(default=1.45, ge=0.0, le=10.0)
    review_severity_threshold: float = Field(default=40.0, ge=0.0, le=100.0)
    fail_severity_threshold: float = Field(default=60.0, ge=0.0, le=100.0)


class ModelMetricsResponse(BaseModel):
    metadata: dict
    runtime_settings: RuntimeModelSettings
    model_comparison: list[dict]
    classifier_report: dict
    confusion_matrix: dict
    threshold_calibration: dict = Field(default_factory=dict)
