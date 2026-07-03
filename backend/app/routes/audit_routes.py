from fastapi import APIRouter, Depends

from app.dependencies import require_roles
from app.models.audit_model import AuditLog
from app.models.user_model import User

router = APIRouter(prefix="/audit-logs", tags=["audit-logs"])


@router.get("")
async def list_audit_logs(
    limit: int = 50,
    _: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> list[dict]:
    limit = max(1, min(limit, 200))
    logs = await AuditLog.find_all().sort("-created_at").limit(limit).to_list()
    return [
        {
            "id": str(log.id),
            "actor_id": log.actor_id,
            "action": log.action,
            "entity_type": log.entity_type,
            "entity_id": log.entity_id,
            "metadata": log.metadata,
            "created_at": log.created_at,
        }
        for log in logs
    ]
