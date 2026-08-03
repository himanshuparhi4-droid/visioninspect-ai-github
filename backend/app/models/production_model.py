from datetime import datetime

from beanie import Document
from pydantic import Field
from pymongo import ASCENDING, IndexModel

from app.time_utils import utc_now


class Product(Document):
    product_id: str
    name: str
    category: str | None = None
    critical_zones: list[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "products"
        indexes = [
            IndexModel([("product_id", ASCENDING)], unique=True, name="product_id_1"),
            "category",
            "is_active",
        ]


class ProductionLine(Document):
    line_id: str
    name: str
    location: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "production_lines"
        indexes = [
            IndexModel([("line_id", ASCENDING)], unique=True, name="line_id_1"),
            "is_active",
        ]


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
        indexes = [
            IndexModel([("batch_number", ASCENDING)], unique=True, name="batch_number_1"),
            "product_id",
            "status",
        ]
