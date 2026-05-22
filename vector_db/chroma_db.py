import contextlib
import json
from pathlib import Path
from typing import Any

from configs.config import settings
from plugin_system.base import BaseVectorStore, Chunk
from utils.logging import logger

chromadb: Any = None
try:
    import chromadb as chromadb_module

    chromadb = chromadb_module
except ImportError:
    pass


class ChromaVectorStore(BaseVectorStore):
    """Local vector store utilizing ChromaDB for similarity search."""

    def __init__(self, db_dir: Path | None = None) -> None:
        self.db_dir = db_dir or settings.CHROMA_DB_PATH
        self._client: Any = None
        self._collection: Any = None

    @property
    def name(self) -> str:
        return "chromadb"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> None:
        if chromadb is None:
            logger.warn(
                "chromadb is not installed. ChromaVectorStore will be unavailable."
            )
            return

        self.db_dir.mkdir(parents=True, exist_ok=True)
        assert chromadb is not None
        try:
            # Persistent client in the db directory
            self._client = chromadb.PersistentClient(path=str(self.db_dir))
            # Create or get collection
            self._collection = self._client.get_or_create_collection(
                name="kchunker_chunks"
            )
            logger.info(
                "ChromaDB persistent client initialized successfully",
                count=self._collection.count(),
            )
        except Exception as e:
            logger.error("Failed to initialize ChromaDB client", error=str(e))

    async def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> bool:
        if self._collection is None:
            logger.error("ChromaDB collection is not initialized. Cannot add chunks.")
            return False

        if not chunks or not embeddings:
            return False

        try:
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [self._sanitize_metadata(chunk) for chunk in chunks]

            self._collection.add(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )
            logger.info("Added chunks to ChromaDB collection", count=len(chunks))
            return True
        except Exception as e:
            logger.error("Failed to add chunks to ChromaDB", error=str(e))
            return False

    async def similarity_search(
        self, query: str, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[Chunk]:
        if self._collection is None:
            logger.warn(
                "ChromaDB collection is not initialized. Returning empty results."
            )
            return []

        # Get embedding model to embed the search query
        from plugin_system.manager import plugin_manager

        embedding_model = plugin_manager.get_embedding_model("sentence_transformers")
        if not embedding_model:
            logger.error("Embedding model not found for query generation")
            return []

        try:
            query_vectors = await embedding_model.embed([query])
            if not query_vectors:
                return []

            # Format filters for ChromaDB
            where_clause = None
            if filters:
                # If one filter, pass directly; if multiple, group under $and
                if len(filters) == 1:
                    k, v = next(iter(filters.items()))
                    where_clause = {k: v}
                else:
                    where_clause = {"$and": [{k: v} for k, v in filters.items()]}

            results = self._collection.query(
                query_embeddings=query_vectors, n_results=limit, where=where_clause
            )

            # Reconstruct Chunk objects
            chunks: list[Chunk] = []
            if not results or not results["ids"] or not results["ids"][0]:
                return []

            for idx in range(len(results["ids"][0])):
                cid = results["ids"][0][idx]
                doc = results["documents"][0][idx]
                meta = results["metadatas"][0][idx]

                chunks.append(self._reconstruct_chunk(cid, doc, meta))

            return chunks
        except Exception as e:
            logger.error("Error executing ChromaDB similarity search", error=str(e))
            return []

    def _sanitize_metadata(self, chunk: Chunk) -> dict[str, Any]:
        """Flattens and serializes complex fields to primitive types."""
        meta: dict[str, Any] = {
            "document_name": chunk.document_name,
            "document_type": chunk.document_type,
            "chunk_type": chunk.chunk_type,
        }
        if chunk.page is not None:
            meta["page"] = chunk.page
        if chunk.section is not None:
            meta["section"] = chunk.section
        if chunk.customer is not None:
            meta["customer"] = chunk.customer
        if chunk.rfq_number is not None:
            meta["rfq_number"] = chunk.rfq_number
        if chunk.revision is not None:
            meta["revision"] = chunk.revision
        if chunk.coordinates:
            meta["coordinates"] = json.dumps(chunk.coordinates)
        if chunk.related_chunks:
            meta["related_chunks"] = json.dumps(chunk.related_chunks)

        # Inject other dictionary metadata
        for k, v in chunk.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = json.dumps(v)

        return meta

    def _reconstruct_chunk(
        self, cid: str, document: str, meta: dict[str, Any]
    ) -> Chunk:
        """Restores full Chunk properties from flat database metadata."""
        coords = []
        if "coordinates" in meta:
            with contextlib.suppress(Exception):
                coords = json.loads(meta["coordinates"])

        related = []
        if "related_chunks" in meta:
            with contextlib.suppress(Exception):
                related = json.loads(meta["related_chunks"])

        # Exclude known fields from raw metadata dict
        extra_meta = {}
        standard_keys = [
            "document_name",
            "document_type",
            "chunk_type",
            "page",
            "section",
            "customer",
            "rfq_number",
            "revision",
            "coordinates",
            "related_chunks",
        ]
        for k, v in meta.items():
            if k not in standard_keys:
                try:
                    # Attempt decoding JSON string values
                    extra_meta[k] = (
                        json.loads(v)
                        if isinstance(v, str) and (v.startswith(("[", "{")))
                        else v
                    )
                except Exception:
                    extra_meta[k] = v

        return Chunk(
            chunk_id=cid,
            content=document,
            document_name=meta.get("document_name", ""),
            document_type=meta.get("document_type", "TXT"),
            chunk_type=meta.get("chunk_type", "TEXT"),
            page=meta.get("page"),
            section=meta.get("section"),
            customer=meta.get("customer"),
            rfq_number=meta.get("rfq_number"),
            revision=meta.get("revision"),
            coordinates=coords,
            related_chunks=related,
            metadata=extra_meta,
        )
