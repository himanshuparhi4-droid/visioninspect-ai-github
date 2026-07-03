from datetime import datetime

from pydantic import BaseModel, Field


class ProductCreate(BaseModel):
    product_id: str = Field(max_length=120)
    name: str = Field(max_length=160)
    category: str = Field(default="Bottle", max_length=120)
    critical_zones: list[str] = Field(default_factory=list)


class ProductResponse(ProductCreate):
    id: str | None = None
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ProductionLineCreate(BaseModel):
    line_id: str = Field(max_length=120)
    name: str = Field(max_length=160)
    location: str | None = Field(default=None, max_length=160)


class ProductionLineResponse(ProductionLineCreate):
    id: str | None = None
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class BatchRecordCreate(BaseModel):
    batch_number: str = Field(max_length=120)
    product_id: str = Field(max_length=120)
    production_line: str = Field(max_length=120)
    shift: str = Field(default="Shift A", max_length=120)
    status: str = Field(default="active", max_length=80)


class BatchRecordResponse(BatchRecordCreate):
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ProductionCatalogResponse(BaseModel):
    products: list[ProductResponse]
    production_lines: list[ProductionLineResponse]
    batches: list[BatchRecordResponse]
    shifts: list[str]
