from beanie import init_beanie

from app.db.mongodb import get_database
from app.models.audit_model import AuditLog
from app.models.inspection_model import Inspection
from app.models.production_model import BatchRecord, Product, ProductionLine
from app.models.report_model import Report
from app.models.rework_model import ReworkTicket
from app.models.user_model import User


async def init_database() -> None:
    await remove_legacy_ttl_indexes()
    await init_beanie(
        database=get_database(),
        document_models=[
            User,
            Inspection,
            Report,
            AuditLog,
            ReworkTicket,
            Product,
            ProductionLine,
            BatchRecord,
        ],
        allow_index_dropping=True,
    )


async def remove_legacy_ttl_indexes() -> None:
    """Remove the old seven-day expiry policy without deleting QA history."""
    database = get_database()
    for collection_name in ("inspections", "batch_records"):
        collection = database[collection_name]
        indexes = await collection.index_information()
        for index_name, definition in indexes.items():
            if definition.get("expireAfterSeconds") is not None:
                await collection.drop_index(index_name)
