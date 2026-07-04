from beanie import init_beanie

from app.db.mongodb import get_database
from app.models.audit_model import AuditLog
from app.models.inspection_model import Inspection
from app.models.model_version_model import ModelVersion
from app.models.production_model import BatchRecord, Product, ProductionLine
from app.models.report_model import Report
from app.models.rework_model import ReworkTicket
from app.models.user_model import User


async def init_database() -> None:
    await init_beanie(
        database=get_database(),
        document_models=[
            User,
            Inspection,
            Report,
            ModelVersion,
            AuditLog,
            ReworkTicket,
            Product,
            ProductionLine,
            BatchRecord,
        ],
    )
