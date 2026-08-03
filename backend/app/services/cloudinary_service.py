import logging
from pathlib import Path
from urllib.parse import unquote, urlparse

import cloudinary
import cloudinary.uploader

from app.config import settings

logger = logging.getLogger(__name__)


class CloudStorageError(RuntimeError):
    pass

cloudinary.config(
    cloud_name=settings.cloudinary_cloud_name,
    api_key=settings.cloudinary_api_key,
    api_secret=settings.cloudinary_api_secret,
    secure=True,
)


def cloudinary_is_configured() -> bool:
    return bool(
        settings.use_cloudinary_storage
        and settings.cloudinary_cloud_name
        and settings.cloudinary_api_key
        and settings.cloudinary_api_secret
    )


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
            timeout=settings.cloudinary_timeout_seconds,
        )
        return str(result["secure_url"])
    except Exception as exc:
        if settings.environment.lower() == "production":
            raise CloudStorageError(
                f"Online image storage did not respond within {settings.cloudinary_timeout_seconds} seconds. "
                "Please retry the inspection."
            ) from exc
        logger.warning("Cloudinary upload failed; using local file URL in development", exc_info=True)
        return local_upload_url(path)


def cloudinary_public_id(url: str | None) -> str | None:
    """Extract a Cloudinary public ID from a secure delivery URL."""
    if not url or "res.cloudinary.com" not in url:
        return None
    path_parts = [unquote(part) for part in urlparse(url).path.split("/") if part]
    try:
        upload_index = path_parts.index("upload")
    except ValueError:
        return None
    asset_parts = path_parts[upload_index + 1 :]
    if asset_parts and asset_parts[0].startswith("v") and asset_parts[0][1:].isdigit():
        asset_parts = asset_parts[1:]
    if not asset_parts:
        return None
    asset_parts[-1] = Path(asset_parts[-1]).stem
    return "/".join(asset_parts)


def cleanup_stored_image(path: str | Path | None, url: str | None = None) -> None:
    """Best-effort rollback for a partially completed inspection."""
    public_id = cloudinary_public_id(url)
    if public_id and cloudinary_is_configured():
        try:
            cloudinary.uploader.destroy(public_id, resource_type="image", invalidate=True)
        except Exception:
            logger.warning("Could not remove rolled-back Cloudinary asset", exc_info=True)

    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove rolled-back local asset %s", path, exc_info=True)
