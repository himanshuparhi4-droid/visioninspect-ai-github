from app.models.production_model import BatchRecord, Product, ProductionLine
from app.schemas.production_schema import BatchRecordResponse, ProductResponse, ProductionCatalogResponse, ProductionLineResponse


DEFAULT_SHIFTS = ["Shift A", "Shift B", "Shift C", "Night Shift"]

DEFAULT_PRODUCTS = [
    ProductResponse(product_id="BOTTLE-STD-500", name="Bottle 500ml", category="Bottle", critical_zones=["neck", "cap-ring", "base"]),
    ProductResponse(product_id="BOTTLE-STD-750", name="Bottle 750ml", category="Bottle", critical_zones=["neck", "body", "base"]),
    ProductResponse(product_id="BOTTLE-QA-SAMPLE", name="QA Sample Bottle", category="Bottle", critical_zones=["body"]),
]

DEFAULT_LINES = [
    ProductionLineResponse(line_id="Line-01", name="Line 01", location="Plant A"),
    ProductionLineResponse(line_id="Line-02", name="Line 02", location="Plant A"),
    ProductionLineResponse(line_id="Line-SIM-01", name="Simulation Line", location="Digital Twin"),
]

DEFAULT_BATCHES = [
    BatchRecordResponse(batch_number="BATCH-DEMO-A", product_id="BOTTLE-STD-500", production_line="Line-01", shift="Shift A"),
    BatchRecordResponse(batch_number="BATCH-DEMO-B", product_id="BOTTLE-STD-750", production_line="Line-02", shift="Shift B"),
    BatchRecordResponse(batch_number="SIM-DEMO", product_id="BOTTLE-QA-SAMPLE", production_line="Line-SIM-01", shift="Shift A"),
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
    products = [product_to_response(item) for item in await Product.find(Product.is_active == True).sort("product_id").to_list()]  # noqa: E712
    lines = [line_to_response(item) for item in await ProductionLine.find(ProductionLine.is_active == True).sort("line_id").to_list()]  # noqa: E712
    batches = [batch_to_response(item) for item in await BatchRecord.find_all().sort("-created_at").limit(100).to_list()]

    return ProductionCatalogResponse(
        products=products or DEFAULT_PRODUCTS,
        production_lines=lines or DEFAULT_LINES,
        batches=batches or DEFAULT_BATCHES,
        shifts=DEFAULT_SHIFTS,
    )
