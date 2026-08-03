from fastapi import HTTPException, status

from app.models.user_model import User
from app.schemas.user_schema import UserCreate
from app.security import create_access_token, hash_password, verify_password
from app.time_utils import utc_now


async def get_user_by_email(email: str) -> User | None:
    return await User.find_one({"email": email.lower()})


async def ensure_bootstrap_admin() -> User | None:
    """Create the documented local admin once for clone-and-run development."""
    from app.config import settings

    if not settings.bootstrap_admin_enabled or settings.environment.lower() in {"production", "prod"}:
        return None

    email = settings.bootstrap_admin_email.lower()
    existing = await get_user_by_email(email)
    if existing is not None:
        return existing

    admin = User(
        name=settings.bootstrap_admin_name.strip(),
        email=email,
        hashed_password=hash_password(settings.bootstrap_admin_password),
        role="admin",
        requested_role="admin",
        approval_status="approved",
        approved_at=utc_now(),
        is_active=True,
        updated_at=utc_now(),
    )
    await admin.insert()
    return admin


async def create_user(
    payload: UserCreate,
    *,
    is_active: bool = True,
    approval_status: str = "approved",
    requested_role: str | None = None,
    approved_by: str | None = None,
) -> User:
    existing_user = await get_user_by_email(payload.email)
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists",
        )

    user = User(
        name=payload.name.strip(),
        email=payload.email.lower(),
        hashed_password=hash_password(payload.password),
        role=payload.role,
        requested_role=requested_role or payload.role,
        approval_status=approval_status,
        approved_by=approved_by,
        approved_at=utc_now() if approval_status == "approved" else None,
        is_active=is_active,
        updated_at=utc_now(),
    )
    await user.insert()
    return user


async def authenticate_user(email: str, password: str) -> User:
    user = await User.find_one({"email": email.lower()})

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if user.approval_status == "pending":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is pending admin approval",
        )
    if user.approval_status == "rejected":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account registration was rejected",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    user.last_login_at = utc_now()
    user.updated_at = utc_now()
    await user.save()
    return user


def build_token_for_user(user: User) -> str:
    return create_access_token(subject=str(user.id))


__all__ = [
    "authenticate_user",
    "build_token_for_user",
    "create_user",
    "ensure_bootstrap_admin",
    "get_user_by_email",
    "hash_password",
    "verify_password",
]
