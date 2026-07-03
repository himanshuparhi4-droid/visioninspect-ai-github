from pydantic import BaseModel


class AnalyticsSummary(BaseModel):
    total_inspections: int
    defective_count: int
    good_count: int
    average_confidence: float

