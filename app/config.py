from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Prefix: FACE_AI_
    Example: FACE_AI_MODEL_PACK_NAME=buffalo_sc
    """

    # Use buffalo_sc (small CPU model pack ~28MB) for lightweight cloud RAM usage < 150MB
    model_pack_name: str = "buffalo_sc"
    detection_size: int = 640
    detection_threshold: float = 0.5
    max_image_dimension: int = 4096
    max_file_size_mb: int = 10
    allowed_origins: list[str] = ["*"]
    log_level: str = "INFO"

    model_config = {"env_prefix": "FACE_AI_"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
