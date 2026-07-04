import logging
from pathlib import Path

import cloudinary
import cloudinary.uploader

from app.config import settings

logger = logging.getLogger(__name__)

cloudinary.config(
    cloud_name=settings.cloudinary_cloud_name,
    api_key=settings.cloudinary_api_key,
    api_secret=settings.cloudinary_api_secret,
    secure=True,
)


def cloudinary_is_configured() -> bool:
    return bool(settings.cloudinary_cloud_name and settings.cloudinary_api_key and settings.cloudinary_api_secret)


def storage_backend() -> str:
    return "cloudinary" if cloudinary_is_configured() else "local"


def local_upload_url(path: str | Path) -> str:
    image_path = Path(path)
    uploads_root = Path(__file__).resolve().parents[1] / "uploads"
    try:
        relative_path = image_path.resolve().relative_to(uploads_root.resolve())
        return f"{settings.backend_url.rstrip('/')}/uploads/{relative_path.as_posix()}"
    except ValueError:
        return str(image_path)


def upload_image_or_local_url(path: str | Path, folder: str) -> str:
    if not cloudinary_is_configured():
        return local_upload_url(path)

    try:
        result = cloudinary.uploader.upload(
            str(path),
            folder=f"visioninspect-ai/{folder}",
            resource_type="auto",
        )
        return str(result["secure_url"])
    except Exception as exc:
        if settings.environment.lower() == "production":
            raise RuntimeError("Cloudinary upload failed in production") from exc
        logger.warning("Cloudinary upload failed; using local file URL in development", exc_info=True)
        return local_upload_url(path)
