import asyncio
import logging
import shutil
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.dependencies import get_current_user, require_roles
from app.models.inspection_model import Inspection
from app.models.production_model import Product
from app.models.user_model import User
from app.schemas.inspection_schema import (
    InspectionListResponse,
    InspectionMetadataUpdate,
    InspectionResponse,
    ReviewStatusUpdate,
)
from app.serializers import inspection_to_response
from app.services.audit_service import record_audit_event
from app.services.cloudinary_service import CloudStorageError, cleanup_stored_image, upload_image_or_local_url
from app.services.prediction_service import PredictionError, inspect_image_file
from app.services.rework_service import create_or_update_rework_ticket
from app.time_utils import utc_now
from app.utils import parse_document_id, uploads_path
from ml.config import MVTEC_DATASET_ROOT
from ml.model_registry import CategoryModelError, category_model_statuses, normalize_category

router = APIRouter(prefix="/inspections", tags=["inspections"])
logger = logging.getLogger(__name__)

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ADMIN_ROLES = {"admin", "quality_manager", "factory_supervisor"}
METADATA_FIELDS = ["batch_number", "product_id", "production_line", "shift", "operator_name", "source_label", "category"]
PREDICTION_FIELDS = [
    "processed_image_url",
    "processed_image_path",
    "heatmap_url",
    "heatmap_path",
    "prediction",
    "defect_type",
    "confidence",
    "detection_confidence",
    "classification_confidence",
    "raw_classification_confidence",
    "classification_confidence_calibrated",
    "subtype_model_status",
    "subtype_model_macro_f1",
    "manual_review_required",
    "detector_engine",
    "detector_kind",
    "classifier_engine",
    "detector_fallback_used",
    "detector_fallback_reason",
    "classifier_fallback_used",
    "classifier_fallback_reason",
    "anomaly_score",
    "defect_area_ratio",
    "severity_score",
    "severity_level",
    "pass_fail",
    "recommended_action",
    "model_version",
]
APP_ROOT = Path(__file__).resolve().parents[1]

def get_camera_sample_root(category: str) -> Path:
    return Path(MVTEC_DATASET_ROOT) / category / "test"

def get_demo_sample_root(category: str) -> Path:
    return APP_ROOT / "demo_samples" / category / "test"

def optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def metadata_fields(metadata: dict | None) -> dict:
    metadata = metadata or {}
    return {field: optional_text(metadata.get(field)) for field in METADATA_FIELDS}


def automatic_metadata(metadata: dict | None, current_user: User, source_label: str | None = None) -> dict:
    values = metadata_fields(metadata)
    try:
        category = normalize_category(values["category"])
    except CategoryModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    now = utc_now()
    source = values["source_label"] or optional_text(source_label) or "uploaded-image"
    stem = "".join(character if character.isalnum() else "-" for character in Path(source).stem).strip("-")
    product_suffix = (stem or uuid4().hex[:8]).upper()[:32]
    return {
        **values,
        "batch_number": values["batch_number"] or f"AUTO-{now:%Y%m%d}",
        "product_id": values["product_id"] or f"{category.upper()}-{product_suffix}",
        "production_line": values["production_line"] or "Line-Manual-01",
        "shift": values["shift"] or "Auto Shift",
        "operator_name": values["operator_name"] or current_user.name,
        "source_label": source,
        "category": category,
    }


async def store_image(path: Path, folder: str) -> str:
    try:
        return await run_in_threadpool(upload_image_or_local_url, path, folder)
    except CloudStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
        "average_confidence": round(sum(item.confidence or 0 for item in inspections) / len(inspections), 4) if inspections else 0.0,
    }


def ordered_label_counts(paths: list[Path]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for path in paths:
        labels[path.parent.name] = labels.get(path.parent.name, 0) + 1
    # Order 'good' first, then the rest
    ordered = {}
    if "good" in labels:
        ordered["good"] = labels["good"]
    ordered.update({label: count for label, count in labels.items() if label != "good"})
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
    category: str | None = Form(default=None),
) -> dict:
    return {
        "batch_number": optional_text(batch_number),
        "product_id": optional_text(product_id),
        "production_line": optional_text(production_line),
        "shift": optional_text(shift),
        "operator_name": optional_text(operator_name),
        "source_label": optional_text(source_label),
        "category": optional_text(category),
    }


async def get_visible_inspection(inspection_id: str, current_user: User) -> Inspection:
    inspection = await Inspection.get(parse_document_id(inspection_id, "inspection"))
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
    request_started_at = perf_counter()
    metadata = automatic_metadata(metadata, current_user)
    original_url = None
    prediction: dict = {}
    try:
        product = await Product.find_one(Product.product_id == metadata["product_id"])
        critical_zones = tuple(product.critical_zones) if product else ()
        original_task = asyncio.create_task(store_image(image_path, "original"))
        prediction_task = asyncio.create_task(
            run_in_threadpool(
                inspect_image_file,
                image_path,
                metadata["category"],
                critical_zones,
            )
        )
        original_result, prediction_result = await asyncio.gather(
            original_task,
            prediction_task,
            return_exceptions=True,
        )
        if not isinstance(original_result, BaseException):
            original_url = original_result
        if not isinstance(prediction_result, BaseException):
            prediction = prediction_result
        for result in (original_result, prediction_result):
            if isinstance(result, BaseException):
                raise result
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
    except CloudStorageError as exc:
        cleanup_stored_image(image_path, original_url)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PredictionError as exc:
        cleanup_stored_image(image_path, original_url)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception:
        cleanup_stored_image(image_path, original_url)
        cleanup_stored_image(prediction.get("processed_image_path"), prediction.get("processed_image_url"))
        cleanup_stored_image(prediction.get("heatmap_path"), prediction.get("heatmap_url"))
        raise
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
            "category": inspection.category,
        },
    )
    logger.info(
        "inspection_request_timing category=%s source_type=%s total_ms=%.1f",
        metadata["category"],
        source_type,
        (perf_counter() - request_started_at) * 1000,
    )
    return inspection


async def create_inspection_from_file(file: UploadFile, current_user: User, metadata: dict | None = None) -> Inspection:
    metadata = {
        **(metadata or {}),
        "source_label": optional_text((metadata or {}).get("source_label")) or file.filename,
    }
    image_path = await save_upload(file)
    return await create_inspection_from_path(image_path, current_user, metadata, source_type="manual_upload")


def camera_sample_paths(category: str, label: str | None = None) -> list[Path]:
    for sample_root in (get_camera_sample_root(category), get_demo_sample_root(category)):
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
    failures = []
    for file in files:
        try:
            inspection = await create_inspection_from_file(
                file,
                current_user,
                {**metadata, "source_label": file.filename},
            )
            inspections.append(inspection)
        except HTTPException as exc:
            failures.append(
                {
                    "file_name": file.filename or "unnamed-image",
                    "status": exc.status_code,
                    "message": str(exc.detail),
                }
            )
        except Exception:
            failures.append(
                {
                    "file_name": file.filename or "unnamed-image",
                    "status": 500,
                    "message": "The image could not be inspected",
                }
            )

    return InspectionListResponse(
        total=len(inspections),
        items=[inspection_to_response(inspection) for inspection in inspections],
        failures=failures,
        summary={
            **summarize_batch(inspections),
            "requested": len(files),
            "succeeded": len(inspections),
            "failed": len(failures),
        },
    )


@router.get("/camera-samples")
async def get_camera_samples(category: str = Query(default="bottle"), current_user: User = Depends(get_current_user)) -> dict:
    paths = camera_sample_paths(category)
    labels = ordered_label_counts(paths)
    await record_audit_event(
        actor=current_user,
        action="camera.samples_viewed",
        entity_type="camera_simulation",
        metadata={"total_samples": len(paths), "labels": labels, "category": category},
    )

    demo_controls = [{"value": "", "label": f"Mixed {category} stream"}]
    for lbl in labels.keys():
        name = "Good items" if lbl == "good" else f"{lbl.replace('_', ' ').title()} defects"
        demo_controls.append({"value": lbl, "label": name})

    return {
        "category": category,
        "total": len(paths),
        "labels": labels,
        "demo_controls": demo_controls,
    }


@router.get("/model-categories")
async def get_model_categories(current_user: User = Depends(get_current_user)) -> dict:
    """Expose portable runtime and camera-sample readiness to the UI."""
    items = []
    for status_item in category_model_statuses(
        settings.use_padim_inference,
        settings.use_openvino_inference,
    ):
        sample_count = len(camera_sample_paths(status_item["category"]))
        items.append(
            {
                **status_item,
                "camera_ready": sample_count > 0,
                "camera_sample_count": sample_count,
            }
        )
    return {"items": items}


@router.post("/camera-simulate", response_model=InspectionResponse, status_code=201)
async def simulate_camera_inspection(
    frame_index: int = Query(default=0, ge=0),
    label: str | None = Query(default=None),
    category: str = Query(default="bottle"),
    current_user: User = Depends(get_current_user),
) -> InspectionResponse:
    paths = camera_sample_paths(category, label)
    if not paths:
        raise HTTPException(status_code=404, detail=f"No camera simulation samples found for category '{category}'")

    sample_path = paths[frame_index % len(paths)]
    image_path = copy_image_to_uploads(sample_path)
    metadata = {
        "batch_number": f"SIM-{utc_now().strftime('%Y%m%d')}",
        "product_id": f"{category.upper()}-STD-500",
        "production_line": "Line-SIM-01",
        "shift": "Simulation",
        "operator_name": "Camera simulator",
        "source_label": f"{sample_path.parent.name}/{sample_path.name}",
        "category": category,
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
    payload: ReviewStatusUpdate,
    current_user: User = Depends(
        require_roles("admin", "quality_manager", "factory_supervisor", "quality_engineer")
    ),
) -> InspectionResponse:
    inspection = await get_visible_inspection(inspection_id, current_user)
    if inspection.review_status in {"approved", "rejected"}:
        raise HTTPException(status_code=409, detail="A finalized inspection cannot be reviewed again")
    next_status = payload.review_status
    if next_status == "sent_for_rework" and not optional_text(payload.review_notes):
        raise HTTPException(status_code=400, detail="Review notes are required when sending an item to rework")
    inspection.review_status = next_status
    inspection.review_notes = optional_text(payload.review_notes)
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
        inspection.rework_ticket_id = str(ticket.id)
        inspection.rework_ticket_number = ticket.ticket_number
        inspection.rework_ticket_status = ticket.status
        await inspection.save()
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
