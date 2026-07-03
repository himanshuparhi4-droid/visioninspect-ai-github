from app.models.audit_model import AuditLog
from app.models.user_model import User


async def record_audit_event(
    *,
    action: str,
    entity_type: str,
    entity_id: str | None = None,
    actor: User | None = None,
    metadata: dict | None = None,
) -> AuditLog:
    log = AuditLog(
        actor_id=str(actor.id) if actor and actor.id else None,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        metadata=metadata or {},
    )
    await log.insert()
    return log
