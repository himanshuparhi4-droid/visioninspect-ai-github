from fastapi import APIRouter, Depends

from app.dependencies import get_current_user, require_roles
from app.models.production_model import BatchRecord, Product, ProductionLine
from app.models.user_model import User
from app.schemas.production_schema import (
    BatchRecordCreate,
    BatchRecordResponse,
    ProductCreate,
    ProductionCatalogResponse,
    ProductionLineCreate,
    ProductionLineResponse,
    ProductResponse,
)
from app.services.audit_service import record_audit_event
from app.services.production_service import (
    batch_to_response,
    build_production_catalog,
    line_to_response,
    product_to_response,
)

router = APIRouter(prefix="/production", tags=["production"])


@router.get("/catalog", response_model=ProductionCatalogResponse)
async def get_production_catalog(current_user: User = Depends(get_current_user)) -> ProductionCatalogResponse:
    return await build_production_catalog()


@router.post("/products", response_model=ProductResponse, status_code=201)
async def create_product(
    payload: ProductCreate,
    current_user: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> ProductResponse:
    existing = await Product.find(Product.product_id == payload.product_id).first_or_none()
    if existing:
        existing.name = payload.name
        existing.category = payload.category
        existing.critical_zones = payload.critical_zones
        existing.is_active = True
        await existing.save()
        item = existing
    else:
        item = Product(**payload.model_dump())
        await item.insert()
    await record_audit_event(
        actor=current_user,
        action="production.product_saved",
        entity_type="product",
        entity_id=str(item.id),
        metadata={"product_id": item.product_id},
    )
    return product_to_response(item)


@router.post("/lines", response_model=ProductionLineResponse, status_code=201)
async def create_production_line(
    payload: ProductionLineCreate,
    current_user: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> ProductionLineResponse:
    existing = await ProductionLine.find(ProductionLine.line_id == payload.line_id).first_or_none()
    if existing:
        existing.name = payload.name
        existing.location = payload.location
        existing.is_active = True
        await existing.save()
        item = existing
    else:
        item = ProductionLine(**payload.model_dump())
        await item.insert()
    await record_audit_event(
        actor=current_user,
        action="production.line_saved",
        entity_type="production_line",
        entity_id=str(item.id),
        metadata={"line_id": item.line_id},
    )
    return line_to_response(item)


@router.post("/batches", response_model=BatchRecordResponse, status_code=201)
async def create_batch_record(
    payload: BatchRecordCreate,
    current_user: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> BatchRecordResponse:
    existing = await BatchRecord.find(BatchRecord.batch_number == payload.batch_number).first_or_none()
    if existing:
        for field, value in payload.model_dump().items():
            setattr(existing, field, value)
        await existing.save()
        item = existing
    else:
        item = BatchRecord(**payload.model_dump())
        await item.insert()
    await record_audit_event(
        actor=current_user,
        action="production.batch_saved",
        entity_type="batch",
        entity_id=str(item.id),
        metadata={"batch_number": item.batch_number},
    )
    return batch_to_response(item)
