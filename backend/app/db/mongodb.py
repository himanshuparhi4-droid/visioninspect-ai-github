from pymongo import AsyncMongoClient

from app.config import settings

client = AsyncMongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)


def get_database():
    return client[settings.mongodb_database]


async def ping_database() -> bool:
    await client.admin.command("ping")
    return True


async def close_database() -> None:
    await client.close()
