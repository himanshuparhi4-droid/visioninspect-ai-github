from fastapi import APIRouter, Request

from app.config import settings
from app.services.cloudinary_service import cloudinary_is_configured, storage_backend
from app.services.prediction_service import resolve_backend_path, uploads_path

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check(request: Request) -> dict:
    checkpoint_path = resolve_backend_path(settings.model_checkpoint_path)
    classifier_path = resolve_backend_path(settings.classifier_model_path)
    reference_path = resolve_backend_path(settings.baseline_reference_path)

    return {
        "status": "ok",
        "environment": settings.environment,
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
        },
    }
