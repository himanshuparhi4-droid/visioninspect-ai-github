from app.models.audit_model import AuditLog
from app.models.user_model import User


async def record_audit_event(
    *,
    action: str,
    entity_type: str,
    entity_id: str | None = None,
    actor_id: str | None = None,
    actor: User | None = None,
    metadata: dict | None = None,
) -> AuditLog | None:
    log = AuditLog(
        actor_id=str(actor.id) if actor and getattr(actor, "id", None) else actor_id,
        actor_name=actor.name if actor else None,
        actor_role=actor.role if actor else None,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        metadata=metadata or {},
    )
    await log.insert()
    return log
