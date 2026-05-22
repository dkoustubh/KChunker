import asyncio
from typing import Any

from configs.config import settings
from plugin_system.base import BaseEmbeddingModel
from utils.logging import logger

SentenceTransformer: Any = None
try:
    from sentence_transformers import SentenceTransformer as ST_Class

    SentenceTransformer = ST_Class
except ImportError:
    pass


class SentenceTransformersModel(BaseEmbeddingModel):
    """Generates vector embeddings using the sentence-transformers library."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.DEFAULT_EMBEDDING_MODEL
        self._model = None

    @property
    def name(self) -> str:
        return "sentence_transformers"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> None:
        if SentenceTransformer is None:
            logger.warn(
                "sentence-transformers is not installed. Embeddings will not be generated."
            )
            return

        try:
            logger.info("Loading sentence-transformers model", model=self.model_name)
            # Load the model and save in embedding cache dir
            self._model = SentenceTransformer(
                self.model_name, cache_folder=str(settings.EMBEDDINGS_CACHE_DIR)
            )
            logger.info("Sentence-transformers model loaded successfully")
        except Exception as e:
            logger.error(
                "Failed to load sentence-transformers model",
                model=self.model_name,
                error=str(e),
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if not self._model:
            logger.warn(
                "Embedding model is not loaded. Returning mock/zero embeddings."
            )
            # Default dimension size of 384 for bge-small
            return [[0.0] * 384 for _ in texts]

        try:
            # sentence-transformers encode is synchronous/blocking, so we run it in executor
            loop = asyncio.get_event_loop()
            # encode returns a numpy array, we convert to list of floats
            embeddings = await loop.run_in_executor(
                None, lambda: self._model.encode(texts, show_progress_bar=False)
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            return [[0.0] * 384 for _ in texts]
