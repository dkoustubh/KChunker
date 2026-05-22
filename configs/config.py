from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    LOG_LEVEL: str = "INFO"
    APP_ENV: str = "production"

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    JSON_STORAGE_DIR: Path = STORAGE_DIR / "json"
    EMBEDDINGS_CACHE_DIR: Path = STORAGE_DIR / "embeddings"
    CACHE_DIR: Path = STORAGE_DIR / "cache"
    VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"

    # Database paths
    CHROMA_DB_PATH: Path = VECTOR_DB_DIR / "chroma"
    FAISS_DB_PATH: Path = VECTOR_DB_DIR / "faiss"

    # OCR Settings
    PADDLEOCR_LANG: str = "en"
    USE_GPU: bool = False

    # Embedding Settings
    DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-small-en"

    # Chunking limits
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    def create_directories(self) -> None:
        """Ensure that storage and cache directories exist on the filesystem."""
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.JSON_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
