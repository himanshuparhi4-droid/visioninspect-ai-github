from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import allowed_cors_origins, settings
from app.db.init_beanie import init_database
from app.db.mongodb import close_database, ping_database
from app.routes import (
    analytics_routes,
    audit_routes,
    auth_routes,
    health_routes,
    inspection_routes,
    model_routes,
    production_routes,
    report_routes,
    rework_routes,
    user_routes,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.database_ready = False
    try:
        await ping_database()
        await init_database()
        app.state.database_ready = True
    except Exception as exc:
        app.state.database_error = str(exc)
    yield
    await close_database()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Manufacturing defect detection and quality inspection API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploads_dir = Path(__file__).resolve().parent / "app" / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

app.include_router(health_routes.router)
app.include_router(auth_routes.router)
app.include_router(user_routes.router)
app.include_router(inspection_routes.router)
app.include_router(analytics_routes.router)
app.include_router(audit_routes.router)
app.include_router(report_routes.router)
app.include_router(rework_routes.router)
app.include_router(production_routes.router)
app.include_router(model_routes.router)


@app.get("/")
async def root() -> dict:
    return {
        "message": "VisionInspect AI backend is running",
        "docs": "/docs",
        "health": "/health",
    }
