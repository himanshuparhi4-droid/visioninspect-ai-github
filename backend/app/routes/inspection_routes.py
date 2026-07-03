import shutil
from pathlib import Path
from uuid import uuid4

from beanie import PydanticObjectId
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_current_user
from app.models.inspection_model import Inspection
from app.models.user_model import User
from app.schemas.inspection_schema import InspectionListResponse, InspectionMetadataUpdate, InspectionResponse, ReviewStatusUpdate
from app.serializers import inspection_to_response
from app.services.cloudinary_service import upload_image_or_local_url
from app.services.audit_service import record_audit_event
from app.services.prediction_service import PredictionError, inspect_image_file, uploads_path
from app.services.rework_service import create_or_update_rework_ticket
from app.time_utils import utc_now

router = APIRouter(prefix="/inspections", tags=["inspections"])

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ADMIN_ROLES = {"admin", "quality_manager", "factory_supervisor"}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAMERA_SAMPLE_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection" / "bottle" / "test"


def parse_document_id(value: str) -> PydanticObjectId:
    try:
        return PydanticObjectId(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid inspection id") from exc


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
        "batch_number": batch_number or None,
        "product_id": product_id or None,
        "production_line": production_line or None,
        "shift": shift or None,
        "operator_name": operator_name or None,
        "source_label": source_label or None,
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
        uploaded_by=str(current_user.id),
        original_image_url=original_url,
        original_image_path=str(image_path),
        processed_image_url=prediction.get("processed_image_url"),
        processed_image_path=prediction.get("processed_image_path"),
        heatmap_url=prediction.get("heatmap_url"),
        heatmap_path=prediction.get("heatmap_path"),
        prediction=prediction.get("prediction"),
        defect_type=prediction.get("defect_type"),
        confidence=prediction.get("confidence"),
        anomaly_score=prediction.get("anomaly_score"),
        defect_area_ratio=prediction.get("defect_area_ratio"),
        class_probabilities=prediction.get("class_probabilities", {}),
        severity_score=prediction.get("severity_score"),
        severity_level=prediction.get("severity_level"),
        severity_components=prediction.get("severity_components", {}),
        explainability=prediction.get("explainability", {}),
        pass_fail=prediction.get("pass_fail"),
        recommended_action=prediction.get("recommended_action"),
        model_version=prediction.get("model_version"),
        review_status="ai_completed",
        batch_number=metadata.get("batch_number"),
        product_id=metadata.get("product_id"),
        production_line=metadata.get("production_line"),
        shift=metadata.get("shift"),
        operator_name=metadata.get("operator_name"),
        source_type=source_type,
        source_label=metadata.get("source_label"),
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
    if not CAMERA_SAMPLE_ROOT.exists():
        return []
    roots = [CAMERA_SAMPLE_ROOT / label] if label else [path for path in CAMERA_SAMPLE_ROOT.iterdir() if path.is_dir()]
    paths: list[Path] = []
    for root in roots:
        if root.exists() and root.is_dir():
            paths.extend(sorted(path for path in root.glob("*.png")))
    return sorted(paths)


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
        uploaded_by=str(current_user.id),
        original_image_url=original_url,
        original_image_path=str(image_path),
        review_status="uploaded",
        batch_number=metadata.get("batch_number"),
        product_id=metadata.get("product_id"),
        production_line=metadata.get("production_line"),
        shift=metadata.get("shift"),
        operator_name=metadata.get("operator_name"),
        source_type="manual_upload",
        source_label=metadata.get("source_label"),
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
        setattr(inspection, field, value.strip() if isinstance(value, str) and value.strip() else None)
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

    summary = {
        "total": len(inspections),
        "good": sum(1 for item in inspections if item.prediction == "Good"),
        "defective": sum(1 for item in inspections if item.prediction == "Defective"),
        "pass": sum(1 for item in inspections if item.pass_fail == "Pass"),
        "review": sum(1 for item in inspections if item.pass_fail == "Review"),
        "fail": sum(1 for item in inspections if item.pass_fail == "Fail"),
        "critical": sum(1 for item in inspections if item.severity_level == "Critical"),
        "average_confidence": round(
            sum(item.confidence or 0 for item in inspections) / len(inspections),
            4,
        ),
    }

    return InspectionListResponse(
        total=len(inspections),
        items=[inspection_to_response(inspection) for inspection in inspections],
        summary=summary,
    )


@router.get("/camera-samples")
async def get_camera_samples(current_user: User = Depends(get_current_user)) -> dict:
    paths = camera_sample_paths()
    labels: dict[str, int] = {}
    for path in paths:
        labels[path.parent.name] = labels.get(path.parent.name, 0) + 1
    await record_audit_event(
        actor=current_user,
        action="camera.samples_viewed",
        entity_type="camera_simulation",
        metadata={"total_samples": len(paths), "labels": labels},
    )
    ordered_labels = {
        key: labels.get(key, 0)
        for key in ["good", "broken_large", "broken_small", "contamination"]
        if labels.get(key, 0)
    }
    ordered_labels.update({key: value for key, value in labels.items() if key not in ordered_labels})
    return {
        "total": len(paths),
        "labels": ordered_labels,
        "demo_controls": [
            {"value": "", "label": "Mixed production stream"},
            {"value": "good", "label": "Good bottles"},
            {"value": "broken_large", "label": "Broken large defects"},
            {"value": "broken_small", "label": "Broken small defects"},
            {"value": "contamination", "label": "Contamination defects"},
        ],
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
        metadata={"review_status": inspection.review_status, "review_notes": inspection.review_notes, **ticket_metadata},
    )
    response = inspection_to_response(inspection)
    response.rework_ticket_id = ticket_metadata.get("rework_ticket_id")
    response.rework_ticket_number = ticket_metadata.get("rework_ticket_number")
    response.rework_ticket_status = ticket_metadata.get("rework_ticket_status")
    return response
