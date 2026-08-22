from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.cloudinary_service import cloudinary_is_configured, storage_backend
from ml.model_registry import category_model_spec, category_model_statuses

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check(request: Request) -> dict:
    category_statuses = category_model_statuses(
        settings.use_padim_inference,
        settings.use_openvino_inference,
    )
    bottle_model = category_model_spec("bottle")
    advanced_enabled = settings.use_padim_inference and any(
        item["advanced_model_available"] for item in category_statuses
    )
    openvino_enabled = settings.use_openvino_inference and any(
        item["active_engine"].endswith("_openvino") for item in category_statuses
    )
    active_engine = (
        "openvino-anomaly-models"
        if openvino_enabled
        else "advanced-anomaly-models"
        if advanced_enabled
        else "opencv-baseline"
    )

    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "started_at": getattr(request.app.state, "started_at", None),
        "database_ready": getattr(request.app.state, "database_ready", False),
        "database_error": getattr(request.app.state, "database_error", None),
        "artifacts": {
            "padim_checkpoint": bottle_model.has_advanced_model,
            "defect_classifier": bottle_model.classifier_path.exists(),
            "baseline_profile": bottle_model.baseline_profile_path.exists(),
            "portable_categories": sum(1 for item in category_statuses if item["available"]),
            "advanced_categories": sum(1 for item in category_statuses if item["advanced_model_available"]),
        },
        "storage": {
            "backend": storage_backend(),
            "cloudinary_configured": cloudinary_is_configured(),
            "local_upload_route": "/uploads",
        },
        "inference": {
            "padim_enabled": advanced_enabled,
            "padim_requested": settings.use_padim_inference,
            "padim_accelerator": settings.padim_inference_accelerator,
            "openvino_enabled": openvino_enabled,
            "openvino_device": settings.openvino_inference_device,
            "active_engine": active_engine,
            "portable_engine": "opencv-baseline",
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
    category_statuses = category_model_statuses(
        settings.use_padim_inference,
        settings.use_openvino_inference,
    )
    checks = {
        "database_ready": getattr(request.app.state, "database_ready", False),
        "portable_detection_models": all(item["available"] for item in category_statuses),
        "defect_classifiers": all(item["classification_trained"] for item in category_statuses),
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
