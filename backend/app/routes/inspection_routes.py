import shutil
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from beanie import PydanticObjectId
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.dependencies import get_current_user
from app.models.inspection_model import Inspection
from app.models.user_model import User
from app.schemas.inspection_schema import (
    InspectionListResponse,
    InspectionMetadataUpdate,
    InspectionResponse,
    ReviewStatusUpdate,
)
from app.serializers import inspection_to_response
from app.services.audit_service import record_audit_event
from app.services.cloudinary_service import upload_image_or_local_url
from app.services.prediction_service import PredictionError, inspect_image_file, uploads_path
from app.services.rework_service import create_or_update_rework_ticket
from app.time_utils import utc_now

router = APIRouter(prefix="/inspections", tags=["inspections"])

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ADMIN_ROLES = {"admin", "quality_manager", "factory_supervisor"}
CAMERA_LABELS = ["good", "broken_large", "broken_small", "contamination"]
CAMERA_DEMO_CONTROLS = [
    {"value": "", "label": "Mixed production stream"},
    {"value": "good", "label": "Good bottles"},
    {"value": "broken_large", "label": "Broken large defects"},
    {"value": "broken_small", "label": "Broken small defects"},
    {"value": "contamination", "label": "Contamination defects"},
]
METADATA_FIELDS = ["batch_number", "product_id", "production_line", "shift", "operator_name", "source_label"]
PREDICTION_FIELDS = [
    "processed_image_url",
    "processed_image_path",
    "heatmap_url",
    "heatmap_path",
    "prediction",
    "defect_type",
    "confidence",
    "anomaly_score",
    "defect_area_ratio",
    "severity_score",
    "severity_level",
    "pass_fail",
    "recommended_action",
    "model_version",
]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
CAMERA_SAMPLE_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection" / "bottle" / "test"
DEMO_SAMPLE_ROOT = APP_ROOT / "demo_samples" / "bottle" / "test"


def parse_document_id(value: str) -> PydanticObjectId:
    try:
        return PydanticObjectId(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid inspection id") from exc


def optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def metadata_fields(metadata: dict | None) -> dict:
    metadata = metadata or {}
    return {field: optional_text(metadata.get(field)) for field in METADATA_FIELDS}


def prediction_fields(prediction: dict) -> dict:
    fields = {field: prediction.get(field) for field in PREDICTION_FIELDS}
    fields["class_probabilities"] = prediction.get("class_probabilities", {})
    fields["severity_components"] = prediction.get("severity_components", {})
    fields["explainability"] = prediction.get("explainability", {})
    return fields


def inspection_base_fields(
    *,
    image_path: Path,
    original_url: str,
    current_user: User,
    metadata: dict | None,
    source_type: str,
) -> dict:
    return {
        "uploaded_by": str(current_user.id),
        "original_image_url": original_url,
        "original_image_path": str(image_path),
        "source_type": source_type,
        **metadata_fields(metadata),
    }


def summarize_batch(inspections: list[Inspection]) -> dict:
    def count(field: str, expected: str) -> int:
        return sum(1 for item in inspections if getattr(item, field) == expected)

    return {
        "total": len(inspections),
        "good": count("prediction", "Good"),
        "defective": count("prediction", "Defective"),
        "pass": count("pass_fail", "Pass"),
        "review": count("pass_fail", "Review"),
        "fail": count("pass_fail", "Fail"),
        "critical": count("severity_level", "Critical"),
        "average_confidence": round(sum(item.confidence or 0 for item in inspections) / len(inspections), 4),
    }


def ordered_label_counts(paths: list[Path]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for path in paths:
        labels[path.parent.name] = labels.get(path.parent.name, 0) + 1
    ordered = {label: labels[label] for label in CAMERA_LABELS if labels.get(label)}
    ordered.update({label: count for label, count in labels.items() if label not in ordered})
    return ordered


async def save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image format",
        )

    destination = uploads_path("original").joinpath(f"{uuid4().hex}{suffix}")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {settings.max_upload_size_mb} MB upload limit",
        )
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    destination.write_bytes(content)
    return destination


def copy_image_to_uploads(source_path: Path) -> Path:
    suffix = source_path.suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    destination = uploads_path("original").joinpath(f"{uuid4().hex}{suffix}")
    shutil.copy2(source_path, destination)
    return destination


def inspection_metadata(
    batch_number: str | None = Form(default=None),
    product_id: str | None = Form(default=None),
    production_line: str | None = Form(default=None),
    shift: str | None = Form(default=None),
    operator_name: str | None = Form(default=None),
    source_label: str | None = Form(default=None),
) -> dict:
    return {
        "batch_number": optional_text(batch_number),
        "product_id": optional_text(product_id),
        "production_line": optional_text(production_line),
        "shift": optional_text(shift),
        "operator_name": optional_text(operator_name),
        "source_label": optional_text(source_label),
    }


async def get_visible_inspection(inspection_id: str, current_user: User) -> Inspection:
    inspection = await Inspection.get(parse_document_id(inspection_id))
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found")

    if current_user.role not in ADMIN_ROLES and inspection.uploaded_by != str(current_user.id):
        raise HTTPException(status_code=403, detail="You cannot access this inspection")
    return inspection


async def create_inspection_from_path(
    image_path: Path,
    current_user: User,
    metadata: dict | None = None,
    source_type: str = "manual_upload",
) -> Inspection:
    metadata = metadata or {}
    original_url = upload_image_or_local_url(image_path, "original")

    try:
        prediction = await run_in_threadpool(inspect_image_file, image_path)
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    inspection = Inspection(
        **inspection_base_fields(
            image_path=image_path,
            original_url=original_url,
            current_user=current_user,
            metadata=metadata,
            source_type=source_type,
        ),
        **prediction_fields(prediction),
        review_status="ai_completed",
    )
    await inspection.insert()
    await record_audit_event(
        actor=current_user,
        action="inspection.completed",
        entity_type="inspection",
        entity_id=str(inspection.id),
        metadata={
            "prediction": inspection.prediction,
            "defect_type": inspection.defect_type,
            "pass_fail": inspection.pass_fail,
            "source_type": source_type,
            "product_id": inspection.product_id,
            "batch_number": inspection.batch_number,
        },
    )
    return inspection


async def create_inspection_from_file(file: UploadFile, current_user: User, metadata: dict | None = None) -> Inspection:
    metadata = {**(metadata or {})}
    metadata["source_label"] = metadata.get("source_label") or file.filename
    image_path = await save_upload(file)
    return await create_inspection_from_path(image_path, current_user, metadata, source_type="manual_upload")


def camera_sample_paths(label: str | None = None) -> list[Path]:
    for sample_root in (CAMERA_SAMPLE_ROOT, DEMO_SAMPLE_ROOT):
        if not sample_root.exists():
            continue
        roots = [sample_root / label] if label else [path for path in sample_root.iterdir() if path.is_dir()]
        paths: list[Path] = []
        for root in roots:
            if root.exists() and root.is_dir():
                paths.extend(sorted(path for path in root.glob("*.png")))
        if paths:
            return sorted(paths)
    return []


@router.post("/upload", response_model=InspectionResponse, status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    metadata: dict = Depends(inspection_metadata),
    current_user: User = Depends(get_current_user),
) -> InspectionResponse:
    metadata = {**metadata}
    metadata["source_label"] = metadata.get("source_label") or file.filename
    image_path = await save_upload(file)
    original_url = upload_image_or_local_url(image_path, "original")
    inspection = Inspection(
        **inspection_base_fields(
            image_path=image_path,
            original_url=original_url,
            current_user=current_user,
            metadata=metadata,
            source_type="manual_upload",
        ),
        review_status="uploaded",
    )
    await inspection.insert()
    await record_audit_event(
        actor=current_user,
        action="inspection.uploaded",
        entity_type="inspection",
        entity_id=str(inspection.id),
        metadata={"product_id": inspection.product_id, "batch_number": inspection.batch_number},
    )
    return inspection_to_response(inspection)


@router.patch("/{inspection_id}/metadata", response_model=InspectionResponse)
async def update_inspection_metadata(
    inspection_id: str,
    payload: InspectionMetadataUpdate,
    current_user: User = Depends(get_current_user),
) -> InspectionResponse:
    inspection = await get_visible_inspection(inspection_id, current_user)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(inspection, field, optional_text(value) if isinstance(value, str) else value)
    inspection.updated_at = utc_now()
    await inspection.save()
    await record_audit_event(
        actor=current_user,
        action="inspection.metadata_updated",
        entity_type="inspection",
        entity_id=str(inspection.id),
        metadata={
            "product_id": inspection.product_id,
            "batch_number": inspection.batch_number,
            "production_line": inspection.production_line,
            "shift": inspection.shift,
        },
    )
    return inspection_to_response(inspection)


@router.post("/inspect", response_model=InspectionResponse, status_code=201)
async def inspect_image(
    file: UploadFile = File(...),
    metadata: dict = Depends(inspection_metadata),
    current_user: User = Depends(get_current_user),
) -> InspectionResponse:
    inspection = await create_inspection_from_file(file, current_user, metadata)
    return inspection_to_response(inspection)


@router.post("/batch-inspect", response_model=InspectionListResponse, status_code=201)
async def batch_inspect_images(
    files: list[UploadFile] = File(...),
    metadata: dict = Depends(inspection_metadata),
    current_user: User = Depends(get_current_user),
) -> InspectionListResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Batch inspection is limited to 20 images")

    inspections = []
    for file in files:
        inspection = await create_inspection_from_file(file, current_user, metadata)
        inspections.append(inspection)

    return InspectionListResponse(
        total=len(inspections),
        items=[inspection_to_response(inspection) for inspection in inspections],
        summary=summarize_batch(inspections),
    )


@router.get("/camera-samples")
async def get_camera_samples(current_user: User = Depends(get_current_user)) -> dict:
    paths = camera_sample_paths()
    labels = ordered_label_counts(paths)
    await record_audit_event(
        actor=current_user,
        action="camera.samples_viewed",
        entity_type="camera_simulation",
        metadata={"total_samples": len(paths), "labels": labels},
    )
    return {
        "total": len(paths),
        "labels": labels,
        "demo_controls": CAMERA_DEMO_CONTROLS,
    }


@router.post("/camera-simulate", response_model=InspectionResponse, status_code=201)
async def simulate_camera_inspection(
    frame_index: int = Query(default=0, ge=0),
    label: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
) -> InspectionResponse:
    paths = camera_sample_paths(label)
    if not paths:
        raise HTTPException(status_code=404, detail="No camera simulation samples found")

    sample_path = paths[frame_index % len(paths)]
    image_path = copy_image_to_uploads(sample_path)
    metadata = {
        "batch_number": f"SIM-{utc_now().strftime('%Y%m%d')}",
        "product_id": f"BOTTLE-{sample_path.stem}",
        "production_line": "Line-SIM-01",
        "shift": "Simulation",
        "operator_name": "Camera simulator",
        "source_label": f"{sample_path.parent.name}/{sample_path.name}",
    }
    inspection = await create_inspection_from_path(
        image_path=image_path,
        current_user=current_user,
        metadata=metadata,
        source_type="camera_simulation",
    )
    await record_audit_event(
        actor=current_user,
        action="camera.frame_inspected",
        entity_type="inspection",
        entity_id=str(inspection.id),
        metadata={"sample": metadata["source_label"], "frame_index": frame_index},
    )
    return inspection_to_response(inspection)


@router.get("", response_model=InspectionListResponse)
async def list_inspections(
    skip: int = 0,
    limit: int = 50,
    product_id: str | None = None,
    production_line: str | None = None,
    review_status: str | None = None,
    current_user: User = Depends(get_current_user),
) -> InspectionListResponse:
    limit = max(1, min(limit, 100))
    filters = []
    if current_user.role not in ADMIN_ROLES:
        filters.append(Inspection.uploaded_by == str(current_user.id))
    if product_id:
        filters.append(Inspection.product_id == product_id)
    if production_line:
        filters.append(Inspection.production_line == production_line)
    if review_status:
        filters.append(Inspection.review_status == review_status)

    query = Inspection.find(*filters) if filters else Inspection.find_all()

    inspections = await query.sort("-created_at").skip(skip).limit(limit).to_list()
    total = await query.count()
    return InspectionListResponse(
        total=total,
        items=[inspection_to_response(inspection) for inspection in inspections],
    )


@router.get("/{inspection_id}", response_model=InspectionResponse)
async def get_inspection(
    inspection_id: str,
    current_user: User = Depends(get_current_user),
) -> InspectionResponse:
    inspection = await get_visible_inspection(inspection_id, current_user)
    return inspection_to_response(inspection)


@router.patch("/{inspection_id}/review-status", response_model=InspectionResponse)
async def update_review_status(
    inspection_id: str,
    payload: ReviewStatusUpdate | None = Body(default=None),
    review_status: str | None = Query(default=None),
    review_notes: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
) -> InspectionResponse:
    inspection = await get_visible_inspection(inspection_id, current_user)
    next_status = payload.review_status if payload else review_status
    if not next_status:
        raise HTTPException(status_code=400, detail="review_status is required")
    inspection.review_status = next_status
    inspection.review_notes = payload.review_notes if payload else review_notes
    inspection.reviewed_by = str(current_user.id)
    inspection.reviewed_at = utc_now()
    inspection.updated_at = utc_now()
    await inspection.save()
    if next_status == "sent_for_rework":
        ticket = await create_or_update_rework_ticket(
            inspection=inspection,
            current_user=current_user,
            reason=inspection.review_notes,
        )
        ticket_metadata = {
            "rework_ticket_id": str(ticket.id),
            "rework_ticket_number": ticket.ticket_number,
            "rework_ticket_status": ticket.status,
        }
    else:
        ticket_metadata = {}
    await record_audit_event(
        actor=current_user,
        action="inspection.review_status_updated",
        entity_type="inspection",
        entity_id=str(inspection.id),
        metadata={
            "review_status": inspection.review_status,
            "review_notes": inspection.review_notes,
            **ticket_metadata,
        },
    )
    response = inspection_to_response(inspection)
    response.rework_ticket_id = ticket_metadata.get("rework_ticket_id")
    response.rework_ticket_number = ticket_metadata.get("rework_ticket_number")
    response.rework_ticket_status = ticket_metadata.get("rework_ticket_status")
    return response
