from app.models.production_model import BatchRecord, Product, ProductionLine
from app.schemas.production_schema import (
    BatchRecordResponse,
    ProductionCatalogResponse,
    ProductionLineResponse,
    ProductResponse,
)
from ml.model_registry import SUPPORTED_CATEGORIES

DEFAULT_SHIFTS = ["Shift A", "Shift B", "Shift C", "Night Shift"]

DEFAULT_PRODUCTS = []
DEFAULT_BATCHES = []

for cat in SUPPORTED_CATEGORIES:
    cat_upper = cat.upper()
    cat_title = cat.replace("_", " ").title()
    DEFAULT_PRODUCTS.extend([
        ProductResponse(
            product_id=f"{cat_upper}-STD-500", name=f"{cat_title} Standard", category=cat, critical_zones=["center", "edge"]
        ),
        ProductResponse(
            product_id=f"{cat_upper}-QA-SAMPLE", name=f"QA Sample {cat_title}", category=cat, critical_zones=["center"]
        ),
    ])
    DEFAULT_BATCHES.extend([
        BatchRecordResponse(
            batch_number=f"BATCH-{cat_upper}-DEMO-A", product_id=f"{cat_upper}-STD-500", production_line="Line-01", shift="Shift A"
        ),
        BatchRecordResponse(
            batch_number=f"SIM-{cat_upper}-DEMO", product_id=f"{cat_upper}-QA-SAMPLE", production_line="Line-SIM-01", shift="Shift A"
        ),
    ])


DEFAULT_LINES = [
    ProductionLineResponse(line_id="Line-01", name="Line 01", location="Plant A"),
    ProductionLineResponse(line_id="Line-02", name="Line 02", location="Plant A"),
    ProductionLineResponse(line_id="Line-SIM-01", name="Simulation Line", location="Digital Twin"),
]


def product_to_response(item: Product) -> ProductResponse:
    return ProductResponse(
        id=str(item.id),
        product_id=item.product_id,
        name=item.name,
        category=item.category,
        critical_zones=item.critical_zones,
        is_active=item.is_active,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def line_to_response(item: ProductionLine) -> ProductionLineResponse:
    return ProductionLineResponse(
        id=str(item.id),
        line_id=item.line_id,
        name=item.name,
        location=item.location,
        is_active=item.is_active,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def batch_to_response(item: BatchRecord) -> BatchRecordResponse:
    return BatchRecordResponse(
        id=str(item.id),
        batch_number=item.batch_number,
        product_id=item.product_id,
        production_line=item.production_line,
        shift=item.shift,
        status=item.status,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


async def build_production_catalog() -> ProductionCatalogResponse:
    products = [
        product_to_response(item) for item in await Product.find({"is_active": True}).sort("product_id").to_list()
    ]
    lines = [
        line_to_response(item)
        for item in await ProductionLine.find({"is_active": True}).sort("line_id").to_list()
    ]
    batches = [
        batch_to_response(item) for item in await BatchRecord.find_all().sort("-created_at").limit(100).to_list()
    ]

    return ProductionCatalogResponse(
        products=products or DEFAULT_PRODUCTS,
        production_lines=lines or DEFAULT_LINES,
        batches=batches or DEFAULT_BATCHES,
        shifts=DEFAULT_SHIFTS,
    )
