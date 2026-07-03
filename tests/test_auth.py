import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.security import create_access_token, decode_access_token, hash_password, verify_password  # noqa: E402


def test_password_hashing_and_verification():
    hashed = hash_password("quality-pass-123")

    assert hashed != "quality-pass-123"
    assert verify_password("quality-pass-123", hashed)
    assert not verify_password("wrong-password", hashed)


def test_jwt_round_trip():
    token = create_access_token("user-id-123")

    assert decode_access_token(token) == "user-id-123"
