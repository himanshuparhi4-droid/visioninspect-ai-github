from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from app.dependencies import get_current_user
from app.models.user_model import User
from app.schemas.user_schema import (
    PublicUserRegister,
    RegistrationResponse,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)
from app.serializers import user_to_response
from app.services.audit_service import record_audit_event
from app.services.auth_service import authenticate_user, build_token_for_user, create_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=RegistrationResponse, status_code=201)
async def register(payload: PublicUserRegister) -> RegistrationResponse:
    user_payload = UserCreate(
        name=payload.name,
        email=payload.email,
        password=payload.password,
        role=payload.role,
    )
    user = await create_user(
        user_payload,
        is_active=False,
        approval_status="pending",
        requested_role=payload.role,
    )
    await record_audit_event(
        actor=user,
        action="auth.registration_requested",
        entity_type="user",
        entity_id=str(user.id),
        metadata={"email": user.email, "requested_role": user.requested_role},
    )
    return RegistrationResponse(
        message="Registration request submitted. An admin must approve the account before login.",
        user=user_to_response(user),
    )


@router.post("/login", response_model=TokenResponse)
async def login(payload: UserLogin) -> TokenResponse:
    user = await authenticate_user(payload.email, payload.password)
    await record_audit_event(
        actor=user,
        action="auth.login",
        entity_type="user",
        entity_id=str(user.id),
        metadata={"email": user.email},
    )
    return TokenResponse(
        access_token=build_token_for_user(user),
        user=user_to_response(user),
    )


@router.post("/token", response_model=TokenResponse)
async def login_with_form(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    user = await authenticate_user(form_data.username, form_data.password)
    await record_audit_event(
        actor=user,
        action="auth.login",
        entity_type="user",
        entity_id=str(user.id),
        metadata={"email": user.email},
    )
    return TokenResponse(
        access_token=build_token_for_user(user),
        user=user_to_response(user),
    )


@router.get("/me", response_model=UserResponse)
async def read_current_user(current_user: User = Depends(get_current_user)) -> UserResponse:
    return user_to_response(current_user)
