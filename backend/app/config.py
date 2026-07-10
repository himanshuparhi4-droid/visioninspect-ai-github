from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VisionInspect AI"
    app_version: str = "1.0.0"
    environment: str = "development"
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 60
    frontend_url: str = "http://localhost:3000"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    backend_url: str = "http://localhost:8000"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "visioninspect_ai"
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""
    cloudinary_timeout_seconds: int = 20
    use_padim_inference: bool = True
    padim_inference_accelerator: str = "auto"
    model_checkpoint_path: str = "../models/checkpoints/padim_mvtec_bottle_v1.ckpt"
    classifier_model_path: str = "../models/defect_classifier.pkl"
    model_metadata_path: str = "../models/model_metadata.json"
    baseline_reference_path: str = "../models/inference/normal_reference.png"
    baseline_threshold: float = 61.99
    upload_dir: str = "app/uploads"
    max_upload_size_mb: int = 10
    request_logging_enabled: bool = True
    security_headers_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def allowed_cors_origins() -> list[str]:
    origins = {origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()}
    origins.add(settings.frontend_url)
    origins.add("https://visioninspect-ai-github.vercel.app")
    return sorted(origins)
