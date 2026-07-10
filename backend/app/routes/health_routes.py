from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.cloudinary_service import cloudinary_is_configured, storage_backend
from app.services.prediction_service import resolve_backend_path, uploads_path

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check(request: Request) -> dict:
    checkpoint_path = resolve_backend_path(settings.model_checkpoint_path)
    classifier_path = resolve_backend_path(settings.classifier_model_path)
    reference_path = resolve_backend_path(settings.baseline_reference_path)
    active_engine = "padim" if settings.use_padim_inference and checkpoint_path.exists() else "baseline"

    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "started_at": getattr(request.app.state, "started_at", None),
        "database_ready": getattr(request.app.state, "database_ready", False),
        "database_error": getattr(request.app.state, "database_error", None),
        "artifacts": {
            "padim_checkpoint": checkpoint_path.exists(),
            "defect_classifier": classifier_path.exists(),
            "baseline_reference": reference_path.exists(),
        },
        "storage": {
            "backend": storage_backend(),
            "cloudinary_configured": cloudinary_is_configured(),
            "local_upload_dir": str(uploads_path()),
        },
        "inference": {
            "padim_enabled": settings.use_padim_inference,
            "padim_accelerator": settings.padim_inference_accelerator,
            "active_engine": active_engine,
            "fallback_engine": "baseline",
        },
    }


@router.get("/live")
async def liveness_check() -> dict:
    return {
        "status": "alive",
        "service": settings.app_name,
        "checked_at": datetime.now(UTC),
    }


@router.get("/ready")
async def readiness_check(request: Request) -> JSONResponse:
    classifier_path = resolve_backend_path(settings.classifier_model_path)
    reference_path = resolve_backend_path(settings.baseline_reference_path)
    checks = {
        "database_ready": getattr(request.app.state, "database_ready", False),
        "defect_classifier": classifier_path.exists(),
        "baseline_reference": reference_path.exists(),
    }
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "checks": checks,
            "database_error": getattr(request.app.state, "database_error", None),
        },
    )
