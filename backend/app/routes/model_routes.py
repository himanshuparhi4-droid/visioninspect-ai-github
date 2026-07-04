from fastapi import APIRouter, Depends

from app.dependencies import get_current_user, require_roles
from app.models.user_model import User
from app.schemas.model_schema import ModelMetricsResponse, RuntimeModelSettings
from app.services.audit_service import record_audit_event
from app.services.model_settings_service import (
    build_model_metrics_payload,
    load_runtime_settings,
    save_runtime_settings,
)

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(current_user: User = Depends(get_current_user)) -> ModelMetricsResponse:
    return ModelMetricsResponse(**build_model_metrics_payload())


@router.get("/settings", response_model=RuntimeModelSettings)
async def get_model_settings(current_user: User = Depends(get_current_user)) -> RuntimeModelSettings:
    return load_runtime_settings()


@router.patch("/settings", response_model=RuntimeModelSettings)
async def update_model_settings(
    payload: RuntimeModelSettings,
    current_user: User = Depends(require_roles("admin", "quality_manager")),
) -> RuntimeModelSettings:
    saved = save_runtime_settings(payload)
    await record_audit_event(
        actor=current_user,
        action="model.settings_updated",
        entity_type="model_settings",
        entity_id="runtime",
        metadata=saved.model_dump(),
    )
    return saved
