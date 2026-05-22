import asyncio
import json
from pathlib import Path

from configs.config import settings
from parsers.router import ParserRouter
from plugin_system.base import Chunk, ExtractedDocument
from plugin_system.manager import plugin_manager
from utils.logging import logger


class WorkflowManager:
    """Orchestrates the ingestion pipeline from raw file to vector storage."""

    def __init__(
        self,
        parser_router: ParserRouter | None = None,
        chunker_name: str = "hybrid_chunker",
        embedding_name: str = "sentence_transformers",
        vector_store_name: str = "chromadb",
    ) -> None:
        self.parser_router = parser_router or ParserRouter()
        self.chunker_name = chunker_name
        self.embedding_name = embedding_name
        self.vector_store_name = vector_store_name

    async def process_file(
        self, file_path: Path, output_dir: Path | None = None
    ) -> list[Chunk]:
        """Ingests, parses, chunks, embeds, and stores a single file."""
        logger.info("Initializing document processing workflow", path=str(file_path))

        # 1. Directory Setup
        settings.create_directories()

        # 2. Parse & Extract
        logger.info("Step 1: Parsing file and extracting document structure")
        extracted_doc: ExtractedDocument = await self.parser_router.route_and_parse(
            file_path
        )
        logger.info("File parsed successfully", doc_type=extracted_doc.document_type)

        # 3. Chunking
        logger.info("Step 2: Performing document chunking")
        chunker = plugin_manager.get_chunker(self.chunker_name)
        if not chunker:
            raise ValueError(f"Chunker plugin '{self.chunker_name}' is not registered.")

        chunks: list[Chunk] = await chunker.chunk(extracted_doc)
        logger.info("Document chunking complete", num_chunks=len(chunks))

        # 4. Save JSON Chunks Locally
        doc_storage_dir = output_dir or (settings.JSON_STORAGE_DIR / file_path.stem)
        doc_storage_dir.mkdir(parents=True, exist_ok=True)

        for idx, chunk in enumerate(chunks):
            chunk_file = doc_storage_dir / f"chunk_{idx + 1}.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(chunk.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(
            "Chunks successfully persisted locally as JSON files",
            target_dir=str(doc_storage_dir),
        )

        if not chunks:
            logger.warn(
                "No chunks generated; skipping embeddings & database storage",
                path=str(file_path),
            )
            return chunks

        # 5. Embedding Generation
        logger.info("Step 3: Generating text embeddings")
        embedding_model = plugin_manager.get_embedding_model(self.embedding_name)
        if not embedding_model:
            raise ValueError(
                f"Embedding model plugin '{self.embedding_name}' is not registered."
            )

        contents = [chunk.content for chunk in chunks]
        embeddings = await embedding_model.embed(contents)

        # Mark chunks as embedded
        for chunk in chunks:
            chunk.embedding_status = True

        logger.info("Embeddings generated successfully", count=len(embeddings))

        # 6. Vector Database Storage
        logger.info("Step 4: Indexing chunks in the vector database")
        vector_store = plugin_manager.get_vector_store(self.vector_store_name)
        if not vector_store:
            raise ValueError(
                f"Vector store plugin '{self.vector_store_name}' is not registered."
            )

        success = await vector_store.add_chunks(chunks, embeddings)
        if success:
            logger.info(
                "Chunks successfully indexed in vector database",
                store=self.vector_store_name,
            )
        else:
            logger.error(
                "Failed to index chunks in vector database",
                store=self.vector_store_name,
            )

        return chunks

    async def process_directory(
        self, dir_path: Path, output_dir: Path | None = None
    ) -> list[Chunk]:
        """Discovers and parses directory files in parallel using a semaphore."""
        logger.info("Processing directory", path=str(dir_path))

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Supported extensions
        extensions = [
            ".pdf",
            ".xlsx",
            ".xls",
            ".csv",
            ".docx",
            ".doc",
            ".eml",
            ".msg",
            ".txt",
            ".md",
            ".json",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
        ]

        files: list[Path] = []
        for ext in extensions:
            files.extend(dir_path.rglob(f"*{ext}"))
            files.extend(dir_path.rglob(f"*{ext.upper()}"))

        # Filter duplicates and remove directories
        files = sorted(
            {fp for fp in files if fp.is_file() and not fp.name.startswith(".")}
        )

        if not files:
            logger.warn("No supported files found in directory", path=str(dir_path))
            return []

        logger.info(
            "Found files to process in directory",
            count=len(files),
            files=[f.name for f in files],
        )

        # Limit concurrency using a semaphore (max 4 concurrent file jobs)
        sem = asyncio.Semaphore(4)
        all_chunks: list[Chunk] = []

        async def sem_process(fp: Path) -> list[Chunk]:
            async with sem:
                try:
                    logger.info("Parallel task starting file processing", path=fp.name)
                    return await self.process_file(fp, output_dir=output_dir)
                except Exception as e:
                    logger.error(
                        "Failed to process file in directory",
                        path=str(fp),
                        error=str(e),
                    )
                    return []

        tasks = [sem_process(fp) for fp in files]
        results = await asyncio.gather(*tasks)

        for res in results:
            all_chunks.extend(res)

        logger.info("Directory processing completed", total_chunks=len(all_chunks))
        return all_chunks
