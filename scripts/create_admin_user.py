import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.init_beanie import init_database  # noqa: E402
from app.db.mongodb import client  # noqa: E402
from app.models.user_model import User  # noqa: E402
from app.security import hash_password  # noqa: E402


async def main() -> None:
    email = os.getenv("ADMIN_EMAIL", "admin@visioninspect.ai").lower()
    password = os.getenv("ADMIN_PASSWORD", "Admin@12345")
    name = os.getenv("ADMIN_NAME", "VisionInspect Admin")

    await client.admin.command("ping")
    await init_database()

    user = await User.find_one(User.email == email)
    if user is None:
        user = User(
            name=name,
            email=email,
            hashed_password=hash_password(password),
            role="admin",
            is_active=True,
        )
        await user.insert()
        print(f"Created admin user: {email}")
    else:
        user.name = name
        user.role = "admin"
        user.is_active = True
        await user.save()
        print(f"Admin user already exists; updated role/status: {email}")

    print("Admin password is configured from ADMIN_PASSWORD or the documented local default.")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
