from pydantic import field_validator
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
    use_cloudinary_storage: bool = False
    use_padim_inference: bool = False
    use_openvino_inference: bool = False
    openvino_inference_device: str = "CPU"
    padim_inference_accelerator: str = "auto"
    model_checkpoint_path: str = "../models/local_checkpoints/padim_mvtec_bottle_v1.ckpt"
    classifier_model_path: str = "../models/defect_classifier.pkl"
    model_metadata_path: str = "../models/model_metadata.json"
    baseline_profile_path: str = "../models/inference/normal_profile.npz"
    baseline_threshold: float = 1.34
    upload_dir: str = "app/uploads"
    max_upload_size_mb: int = 10
    request_logging_enabled: bool = True
    security_headers_enabled: bool = True
    bootstrap_admin_enabled: bool = True
    bootstrap_admin_email: str = "admin@visioninspect.ai"
    bootstrap_admin_password: str = "Admin@12345"
    bootstrap_admin_name: str = "VisionInspect Admin"

    @field_validator("baseline_threshold", mode="before")
    @classmethod
    def migrate_legacy_baseline_threshold(cls, value: object) -> object:
        """Translate the old raw-pixel threshold when upgrading local installs."""
        try:
            numeric_value = float(value)
            return 1.34 if numeric_value > 10 or abs(numeric_value - 1.45) < 1e-9 else value
        except (TypeError, ValueError):
            return value

    @field_validator("model_checkpoint_path", mode="before")
    @classmethod
    def migrate_legacy_bottle_checkpoint_path(cls, value: object) -> object:
        normalized = str(value or "").replace("\\", "/")
        if normalized.endswith("models/checkpoints/padim_mvtec_bottle_v1.ckpt"):
            return "../models/local_checkpoints/padim_mvtec_bottle_v1.ckpt"
        return value

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def allowed_cors_origins() -> list[str]:
    origins = {origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()}
    origins.add(settings.frontend_url)
    return sorted(origins)
