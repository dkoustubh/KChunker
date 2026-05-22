import argparse
import asyncio
import sys
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from chunkers.hybrid import HybridChunker
from configs.config import settings
from embeddings.sentence_transformers import SentenceTransformersModel
from parsers.docx_parser import DOCXParser
from parsers.email_parser import EmailParser
from parsers.excel_parser import ExcelParser
from parsers.ocr_parser import OCRParser

# Import real plugins
from parsers.pdf_parser import PDFParser
from parsers.txt_parser import TXTParser
from plugin_system.base import Chunk
from plugin_system.manager import plugin_manager
from utils.logging import configure_logging, logger
from vector_db.chroma_db import ChromaVectorStore
from vector_db.faiss_db import FAISSVectorStore
from workflows.manager import WorkflowManager


async def animate_spinner(message: str, stop_event: asyncio.Event) -> None:
    """Displays a modern CLI loading animation while processing."""
    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    # Clean visual line start
    sys.stdout.write("\n")
    while not stop_event.is_set():
        sys.stdout.write(
            f"\r\033[94m{spinner_chars[idx % len(spinner_chars)]}\033[0m {message}"
        )
        sys.stdout.flush()
        idx += 1
        await asyncio.sleep(0.08)
    sys.stdout.write(f"\r\033[92m✓\033[0m {message} ... Completed!\n")
    sys.stdout.flush()


async def run_with_spinner(message: str, coro: Coroutine[Any, Any, Any]) -> Any:
    stop_event = asyncio.Event()
    spinner_task = asyncio.create_task(animate_spinner(message, stop_event))
    try:
        result = await coro
        return result
    finally:
        stop_event.set()
        await spinner_task


async def main_async() -> None:
    # 1. CLI Arguments
    parser = argparse.ArgumentParser(
        description="KChunker - Terminal-first layout-aware document chunking and indexing engine."
    )

    # Ingestion group (either file or dir is required for ingestion, unless query is run on existing DB)
    ingest_group = parser.add_mutually_exclusive_group(required=False)
    ingest_group.add_argument(
        "--file", type=str, help="Path to a single document to ingest and process"
    )
    ingest_group.add_argument(
        "--dir", type=str, help="Path to a directory of documents to ingest and process"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Similarity search query to run against the vector store",
    )
    parser.add_argument(
        "--db",
        type=str,
        choices=["chromadb", "faiss"],
        default="chromadb",
        help="Vector database backend to use (default: chromadb)",
    )
    parser.add_argument(
        "--chunk-size", type=int, help="Override default chunk size (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, help="Override default chunk overlap (default: 50)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARN, ERROR)",
    )
    parser.add_argument(
        "--gui",
        "-g",
        action="store_true",
        help="Launch the Dear PyGui management dashboard",
    )

    args = parser.parse_args()

    if args.gui:
        from gui import main_gui
        await main_gui(auto_file=args.file)
        return


    # Apply configuration overrides
    settings.LOG_LEVEL = args.log_level
    configure_logging()

    if args.chunk_size:
        settings.DEFAULT_CHUNK_SIZE = args.chunk_size
    if args.chunk_overlap:
        settings.DEFAULT_CHUNK_OVERLAP = args.chunk_overlap

    logger.info("Initializing KChunker production services")

    # 2. Register Production Plugins
    plugin_manager.register_parser(PDFParser())
    plugin_manager.register_parser(ExcelParser())
    plugin_manager.register_parser(DOCXParser())
    plugin_manager.register_parser(EmailParser())
    plugin_manager.register_parser(TXTParser())

    # Initialize and register OCR parser
    ocr_parser = OCRParser()
    plugin_manager.register_parser(ocr_parser)

    # Register core processors
    plugin_manager.register_chunker(HybridChunker())
    plugin_manager.register_embedding_model(SentenceTransformersModel())
    plugin_manager.register_vector_store(FAISSVectorStore())
    plugin_manager.register_vector_store(ChromaVectorStore())

    # 3. Execution Orchestrator
    manager = WorkflowManager(
        chunker_name="hybrid_chunker",
        embedding_name="sentence_transformers",
        vector_store_name=args.db,
    )

    try:
        # A. INGESTION PHASE
        if args.file or args.dir:
            start_time = time.time()
            chunks: list[Chunk] = []

            if args.file:
                file_path = Path(args.file)
                if not file_path.exists():
                    print(
                        f"\n\033[91m[✗] Error: File does not exist: {file_path}\033[0m"
                    )
                    return

                chunks = await run_with_spinner(
                    f"Processing document: {file_path.name}",
                    manager.process_file(file_path),
                )

            elif args.dir:
                dir_path = Path(args.dir)
                if not dir_path.exists():
                    print(
                        f"\n\033[91m[✗] Error: Directory does not exist: {dir_path}\033[0m"
                    )
                    return

                chunks = await run_with_spinner(
                    f"Ingesting directory: {dir_path.name}",
                    manager.process_directory(dir_path),
                )

            duration = time.time() - start_time
            print(
                f"\n\033[92m[✓] Ingestion successfully completed in {duration:.2f} seconds!\033[0m"
            )
            print(f"    Total Chunks Indexed: {len(chunks)}")

        # B. QUERY PHASE
        if args.query:
            print(f'\n\033[94mRunning Similarity Search Query:\033[0m "{args.query}"')
            vector_store = plugin_manager.get_vector_store(args.db)
            if not vector_store:
                print(
                    f"\033[91m[✗] Error: Vector store '{args.db}' is not registered.\033[0m"
                )
                return

            results = await run_with_spinner(
                f"Querying vector store ({args.db})",
                vector_store.similarity_search(args.query, limit=4),
            )

            if not results:
                print("\n\033[93m[!] No matching chunks found in database.\033[0m")
            else:
                print(f"\n\033[92mFound {len(results)} matching chunks:\033[0m\n")
                for i, r in enumerate(results):
                    source_info = f"{r.document_name}"
                    if r.page:
                        source_info += f" | Page {r.page}"
                    if r.section:
                        source_info += f" | Section: {r.section}"
                    if r.rfq_number:
                        source_info += f" | RFQ: {r.rfq_number}"

                    print(
                        f"\033[1;33m[Result {i + 1}] Source: {source_info} "
                        f"(ID: {r.chunk_id})\033[0m"
                    )
                    print("-" * 80)
                    print(r.content)
                    print("-" * 80 + "\n")

        # C. IF NEITHER RUN INSTRUCTIONS
        if not (args.file or args.dir or args.query):
            parser.print_help()

    except Exception as e:
        logger.exception("An error occurred during workflow execution", error=str(e))
        print(f"\n\033[91m[✗] Workflow failed: {e}\033[0m")
    finally:
        # Shutdown plugins
        plugin_manager.shutdown()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
