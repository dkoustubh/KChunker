import asyncio
import json
from pathlib import Path
from typing import Any

from configs.config import settings
from plugin_system.base import BaseVectorStore, Chunk
from utils.logging import logger

faiss: Any = None
np: Any = None
try:
    import faiss as faiss_module
    import numpy as np_module

    faiss = faiss_module
    np = np_module
except ImportError:
    pass


class FAISSVectorStore(BaseVectorStore):
    """Local vector store utilizing FAISS for fast similarity search."""

    def __init__(self, db_dir: Path | None = None) -> None:
        self.db_dir = db_dir or settings.FAISS_DB_PATH
        self.index_file = self.db_dir / "faiss_index.index"
        self.metadata_file = self.db_dir / "faiss_metadata.json"
        self._index = None
        self._chunks: list[Chunk] = []

    @property
    def name(self) -> str:
        return "faiss"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> None:
        if faiss is None or np is None:
            logger.warn(
                "FAISS or numpy is not installed. FAISSVectorStore will be unavailable."
            )
            return

        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Load existing index if it exists
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                self._index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, encoding="utf-8") as f:
                    meta_list = json.load(f)
                    self._chunks = [Chunk(**c) for c in meta_list]
                logger.info(
                    "Loaded existing FAISS index and metadata", count=len(self._chunks)
                )
            except Exception as e:
                logger.error("Failed to load existing FAISS index", error=str(e))
                self._index = None
                self._chunks = []

    def _save_to_disk(self) -> None:
        if not self._index:
            return
        try:
            faiss.write_index(self._index, str(self.index_file))
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(
                    [c.model_dump() for c in self._chunks],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.debug("Persisted FAISS index and metadata to disk")
        except Exception as e:
            logger.error("Failed to persist FAISS index to disk", error=str(e))

    async def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> bool:
        if faiss is None or np is None:
            logger.error("FAISS or numpy is not installed. Cannot add chunks.")
            return False

        if not chunks or not embeddings:
            return False

        try:
            dim = len(embeddings[0])
            np_embeddings = np.array(embeddings, dtype=np.float32)

            # Lazy initialize the index
            if self._index is None:
                self._index = faiss.IndexFlatL2(dim)
                logger.info("Initialized new FAISS index", dimension=dim)

            # Add to index
            assert self._index is not None
            self._index.add(np_embeddings)
            self._chunks.extend(chunks)

            # Save to disk
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_to_disk)
            return True
        except Exception as e:
            logger.error("Failed to add chunks to FAISS index", error=str(e))
            return False

    async def similarity_search(
        self, query: str, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[Chunk]:
        if faiss is None or np is None or self._index is None or not self._chunks:
            logger.warn(
                "FAISS index is empty or uninitialized. Returning empty search results."
            )
            return []

        # Generate query embedding
        from plugin_system.manager import plugin_manager

        # We assume the sentence_transformers plugin is registered
        embedding_model = plugin_manager.get_embedding_model("sentence_transformers")
        if not embedding_model:
            logger.error("Embedding model not found for query generation")
            return []

        try:
            query_vectors = await embedding_model.embed([query])
            if not query_vectors:
                return []

            np_query = np.array(query_vectors, dtype=np.float32)

            # Run similarity search
            # Search for more than the limit if filters are applied for post-filtering
            search_limit = limit * 4 if filters else limit
            search_limit = min(search_limit, len(self._chunks))

            distances, indices = self._index.search(np_query, search_limit)

            results: list[Chunk] = []
            for _, idx in zip(distances[0], indices[0], strict=True):
                if idx < 0 or idx >= len(self._chunks):
                    continue

                chunk = self._chunks[idx]

                # Check filters
                if filters:
                    match = True
                    for key, val in filters.items():
                        # Try matching top-level properties or metadata fields
                        chunk_val = getattr(chunk, key, None)
                        if chunk_val is None:
                            chunk_val = chunk.metadata.get(key)

                        if chunk_val != val:
                            match = False
                            break
                    if not match:
                        continue

                results.append(chunk)
                if len(results) >= limit:
                    break

            return results
        except Exception as e:
            logger.error("Error executing FAISS similarity search", error=str(e))
            return []
