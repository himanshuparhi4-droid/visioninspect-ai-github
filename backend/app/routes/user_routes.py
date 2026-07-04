from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_current_user, require_roles
from app.models.user_model import User
from app.schemas.user_schema import PasswordReset, UserCreate, UserResponse, UserUpdate
from app.security import hash_password
from app.serializers import user_to_response
from app.services.audit_service import record_audit_event
from app.services.auth_service import create_user
from app.time_utils import utc_now

router = APIRouter(prefix="/users", tags=["users"])

CREATABLE_ROLES = {
    "admin": {"admin", "quality_manager", "factory_supervisor", "quality_engineer"},
    "quality_manager": {"factory_supervisor", "quality_engineer"},
    "factory_supervisor": {"quality_engineer"},
}


def parse_user_id(value: str) -> PydanticObjectId:
    try:
        return PydanticObjectId(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid user id") from exc


async def get_managed_user(user_id: str) -> User:
    user = await User.get(parse_user_id(user_id))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def ensure_can_assign_role(actor: User, role: str | None) -> None:
    if role is None:
        return
    allowed_roles = CREATABLE_ROLES.get(actor.role, set())
    if role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot assign that role",
        )


def ensure_can_manage_user(actor: User, target: User) -> None:
    if actor.role == "admin":
        return
    if target.role == "admin" or target.role == "quality_manager":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot manage this account",
        )
    if actor.role == "factory_supervisor" and target.role != "quality_engineer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot manage this account",
        )


@router.get("/me", response_model=UserResponse)
async def get_profile(current_user: User = Depends(get_current_user)) -> UserResponse:
    return user_to_response(current_user)


@router.get("", response_model=list[UserResponse])
async def list_users(
    _: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> list[UserResponse]:
    users = await User.find_all().sort("-created_at").to_list()
    return [user_to_response(user) for user in users]


@router.post("", response_model=UserResponse, status_code=201)
async def create_managed_user(
    payload: UserCreate,
    current_user: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> UserResponse:
    allowed_roles = CREATABLE_ROLES.get(current_user.role, set())
    if payload.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot create an account with that role",
        )

    user = await create_user(payload, approved_by=str(current_user.id))
    await record_audit_event(
        actor=current_user,
        action="user.created",
        entity_type="user",
        entity_id=str(user.id),
        metadata={"email": user.email, "role": user.role},
    )
    return user_to_response(user)


@router.post("/{user_id}/approve", response_model=UserResponse)
async def approve_registration_request(
    user_id: str,
    current_user: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> UserResponse:
    user = await get_managed_user(user_id)
    ensure_can_manage_user(current_user, user)
    requested_role = user.requested_role or user.role
    ensure_can_assign_role(current_user, requested_role)

    user.role = requested_role
    user.requested_role = requested_role
    user.approval_status = "approved"
    user.is_active = True
    user.approved_by = str(current_user.id)
    user.approved_at = utc_now()
    user.updated_at = utc_now()
    await user.save()
    await record_audit_event(
        actor=current_user,
        action="user.registration_approved",
        entity_type="user",
        entity_id=str(user.id),
        metadata={"email": user.email, "role": user.role},
    )
    return user_to_response(user)


@router.post("/{user_id}/reject", response_model=UserResponse)
async def reject_registration_request(
    user_id: str,
    current_user: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> UserResponse:
    user = await get_managed_user(user_id)
    ensure_can_manage_user(current_user, user)

    user.approval_status = "rejected"
    user.is_active = False
    user.approved_by = str(current_user.id)
    user.approved_at = utc_now()
    user.updated_at = utc_now()
    await user.save()
    await record_audit_event(
        actor=current_user,
        action="user.registration_rejected",
        entity_type="user",
        entity_id=str(user.id),
        metadata={"email": user.email, "requested_role": user.requested_role},
    )
    return user_to_response(user)


@router.patch("/{user_id}", response_model=UserResponse)
async def update_managed_user(
    user_id: str,
    payload: UserUpdate,
    current_user: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> UserResponse:
    user = await get_managed_user(user_id)
    ensure_can_manage_user(current_user, user)
    ensure_can_assign_role(current_user, payload.role)

    if payload.name is not None:
        user.name = payload.name.strip()
    if payload.role is not None:
        user.role = payload.role
        user.requested_role = payload.role
    if payload.is_active is not None:
        if str(user.id) == str(current_user.id) and payload.is_active is False:
            raise HTTPException(status_code=400, detail="You cannot deactivate your own account")
        user.is_active = payload.is_active
        if payload.is_active and user.approval_status != "approved":
            user.approval_status = "approved"
            user.approved_by = str(current_user.id)
            user.approved_at = utc_now()

    user.updated_at = utc_now()
    await user.save()
    await record_audit_event(
        actor=current_user,
        action="user.updated",
        entity_type="user",
        entity_id=str(user.id),
        metadata={"email": user.email, "role": user.role, "is_active": user.is_active},
    )
    return user_to_response(user)


@router.post("/{user_id}/reset-password", response_model=UserResponse)
async def reset_managed_user_password(
    user_id: str,
    payload: PasswordReset,
    current_user: User = Depends(require_roles("admin", "quality_manager", "factory_supervisor")),
) -> UserResponse:
    user = await get_managed_user(user_id)
    ensure_can_manage_user(current_user, user)
    user.hashed_password = hash_password(payload.password)
    user.updated_at = utc_now()
    await user.save()
    await record_audit_event(
        actor=current_user,
        action="user.password_reset",
        entity_type="user",
        entity_id=str(user.id),
        metadata={"email": user.email},
    )
    return user_to_response(user)
