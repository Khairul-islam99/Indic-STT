# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Hugging Face token (required for some gated models, optional here)
    HF_TOKEN: str | None = None

    # Model configuration
    MODEL_NAME: str = "ai4bharat/indic-conformer-600m-multilingual"
    DEVICE: str = "cuda"  # "cuda" or "cpu"
    DECODING_METHOD: str = "rnnt"  # "rnnt" is best for Bengali
    SAMPLE_RATE: int = 16000

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

settings = Settings()