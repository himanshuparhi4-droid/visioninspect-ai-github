from pathlib import Path

from beanie import PydanticObjectId
from fastapi import HTTPException

from app.config import settings

BACKEND_DIR = Path(__file__).resolve().parents[1]


def resolve_backend_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (BACKEND_DIR / path).resolve()


def uploads_path(*parts: str, create: bool = True) -> Path:
    path = resolve_backend_path(settings.upload_dir).joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def parse_document_id(value: str, label: str = "document") -> PydanticObjectId:
    try:
        return PydanticObjectId(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {label} id") from exc
