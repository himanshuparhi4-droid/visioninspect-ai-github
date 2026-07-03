import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.init_beanie import init_database  # noqa: E402
from app.db.mongodb import client, get_database  # noqa: E402
from app.models.model_version_model import ModelVersion  # noqa: E402


async def main() -> None:
    await client.admin.command("ping")
    await init_database()

    metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = metadata.get("model_name", "padim")
    model_version = metadata.get("model_version", "v1")

    model = await ModelVersion.find_one(
        ModelVersion.name == model_name,
        ModelVersion.version == model_version,
    )
    if model is None:
        model = ModelVersion(
            name=model_name,
            version=model_version,
            artifact_path=metadata.get("checkpoint_path", ""),
            metrics=metadata.get("metrics", {}),
        )
        await model.insert()
        print(f"Created model version: {model_name}:{model_version}")
    else:
        model.artifact_path = metadata.get("checkpoint_path", model.artifact_path)
        model.metrics = metadata.get("metrics", {})
        await model.save()
        print(f"Updated model version: {model_name}:{model_version}")

    collections = await get_database().list_collection_names()
    print(f"Database collections: {', '.join(sorted(collections))}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
