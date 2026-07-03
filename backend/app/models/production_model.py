from datetime import datetime

from beanie import Document
from pydantic import Field

from app.time_utils import utc_now


class Product(Document):
    product_id: str
    name: str
    category: str = "Bottle"
    critical_zones: list[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "products"


class ProductionLine(Document):
    line_id: str
    name: str
    location: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "production_lines"


class BatchRecord(Document):
    batch_number: str
    product_id: str
    production_line: str
    shift: str = "Shift A"
    status: str = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "batch_records"
