import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.db.init_beanie import init_database  # noqa: E402
from app.db.mongodb import client, get_database  # noqa: E402
from app.models.user_model import User  # noqa: E402
from app.security import hash_password  # noqa: E402


async def main() -> None:
    await client.admin.command("ping")
    await init_database()

    admin_email = settings.bootstrap_admin_email.lower()
    admin = await User.find_one(User.email == admin_email)
    if admin is None:
        admin = User(
            name=settings.bootstrap_admin_name,
            email=admin_email,
            hashed_password=hash_password(settings.bootstrap_admin_password),
            role="admin",
            requested_role="admin",
            approval_status="approved",
            is_active=True,
        )
        await admin.insert()
        print(f"Created development admin user: {admin_email}")
    else:
        print(f"Development admin already exists: {admin_email}")

    collections = await get_database().list_collection_names()
    print(f"Database collections: {', '.join(sorted(collections))}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
