from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ----------------------------------------------------
# 1. Base Data Models
# ----------------------------------------------------


class Chunk(BaseModel):
    """Represents a standardized document chunk with metadata and relationships.

    Contains structural positioning, source metadata, content, and references.
    """

    chunk_id: str
    document_name: str
    document_type: str
    chunk_type: str = Field(
        description=(
            "TEXT, TABLE, EMAIL, IMAGE_OCR, TECHNICAL_SPEC, BOM, "
            "DRAWING_ANNOTATION"
        )
    )
    page: int | None = None
    section: str | None = None
    customer: str | None = None
    rfq_number: str | None = None
    revision: str | None = None
    content: str
    coordinates: list[Any] = Field(
        default_factory=list, description="Bounding boxes or spatial dimensions"
    )
    related_chunks: list[str] = Field(
        default_factory=list, description="IDs of linked/related chunks"
    )
    embedding_status: bool = False
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary additional key-value pairs"
    )


class ExtractedDocument(BaseModel):
    """Parsed document container capturing structure and extracted components.

    Holds raw text, pages, tables, emails, and metadata.
    """

    document_name: str
    document_path: str
    document_type: str = Field(description="PDF, EXCEL, DOCX, TXT, EMAIL, IMAGE")
    raw_text: str
    pages: list[dict[str, Any]] = Field(
        default_factory=list, description="Page-specific structure, e.g. text and bbox"
    )
    tables: list[dict[str, Any]] = Field(
        default_factory=list, description="Extracted structured tables"
    )
    emails: list[dict[str, Any]] = Field(
        default_factory=list, description="Parsed email chains / histories"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------
# 2. System Interfaces (Plugins, Parsers, Chunkers, DBs)
# ----------------------------------------------------


class BasePlugin(ABC):
    """Base class for all system plugins to manage lifecycle and dynamic discovery."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier/name for the plugin."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version of the plugin."""
        pass

    def initialize(self) -> None:  # noqa: B027
        """Optional lifecycle hook called when initializing the plugin."""
        pass

    def shutdown(self) -> None:  # noqa: B027
        """Optional lifecycle hook called on application shutdown."""
        pass


class BaseParser(BasePlugin):
    """Abstract base class for document parsers."""

    @abstractmethod
    def can_parse(self, file_path: Path, mime_type: str) -> bool:
        """Returns True if this parser can handle the given file."""
        pass

    @abstractmethod
    async def parse(self, file_path: Path) -> ExtractedDocument:
        """Asynchronously parses a document file into an ExtractedDocument container."""
        pass


class BaseChunker(BasePlugin):
    """Abstract base class for chunking engines."""

    @abstractmethod
    async def chunk(self, doc: ExtractedDocument) -> list[Chunk]:
        """Asynchronously converts an ExtractedDocument into structured chunks."""
        pass


class BaseEmbeddingModel(BasePlugin):
    """Abstract base class for embedding generators."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously generates high-dimensional vectors for a list of texts."""
        pass


class BaseVectorStore(BasePlugin):
    """Abstract base class for vector database storage and search operations."""

    @abstractmethod
    async def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> bool:
        """Asynchronously persists chunks and their embeddings into the store."""
        pass

    @abstractmethod
    async def similarity_search(
        self, query: str, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[Chunk]:
        """Asynchronously queries the store for matches based on vector distance."""
        pass
