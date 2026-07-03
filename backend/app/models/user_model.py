from datetime import datetime

from beanie import Document
from pydantic import EmailStr, Field
from pymongo import ASCENDING, IndexModel

from app.time_utils import utc_now


class User(Document):
    name: str
    email: EmailStr
    hashed_password: str
    role: str = Field(default="quality_engineer")
    requested_role: str | None = None
    approval_status: str = "approved"
    approved_by: str | None = None
    approved_at: datetime | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_login_at: datetime | None = None

    class Settings:
        name = "users"
        indexes = [
            IndexModel([("email", ASCENDING)], unique=True),
        ]
