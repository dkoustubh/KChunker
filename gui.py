import asyncio
import json
import time
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg

from chunkers.hybrid import HybridChunker
from configs.config import settings
from embeddings.sentence_transformers import SentenceTransformersModel
from parsers.docx_parser import DOCXParser
from parsers.email_parser import EmailParser
from parsers.excel_parser import ExcelParser
from parsers.ocr_parser import OCRParser
from parsers.pdf_parser import PDFParser
from parsers.txt_parser import TXTParser
from plugin_system.manager import plugin_manager
from utils.logging import configure_logging, logger
from vector_db.chroma_db import ChromaVectorStore
from vector_db.faiss_db import FAISSVectorStore
from workflows.manager import WorkflowManager


# -------------------------------------------------------------
# Global Plugin Registration Helper
# -------------------------------------------------------------
def register_all_plugins() -> None:
    """Registers all production plugins with the global plugin manager."""
    logger.info("Registering all production plugins for GUI execution")
    plugin_manager.register_parser(PDFParser())
    plugin_manager.register_parser(ExcelParser())
    plugin_manager.register_parser(DOCXParser())
    plugin_manager.register_parser(EmailParser())
    plugin_manager.register_parser(TXTParser())

    ocr_parser = OCRParser()
    plugin_manager.register_parser(ocr_parser)

    plugin_manager.register_chunker(HybridChunker())
    plugin_manager.register_embedding_model(SentenceTransformersModel())
    plugin_manager.register_vector_store(FAISSVectorStore())
    plugin_manager.register_vector_store(ChromaVectorStore())


def get_store_chunk_count(store_name: str) -> int:
    """Retrieves the current number of indexed chunks from the vector store."""
    store = plugin_manager.get_vector_store(store_name)
    if not store:
        return 0
    if store_name == "chromadb" and isinstance(store, ChromaVectorStore):
        if store._collection is not None:
            try:
                count: int = store._collection.count()
                return count
            except Exception:
                return 0
    elif store_name == "faiss" and isinstance(store, FAISSVectorStore):
        return len(store._chunks)
    return 0


# -------------------------------------------------------------
# Callbacks and Async Engine Wrappers
# -------------------------------------------------------------
async def run_ingest(
    file_path_str: str, db_backend: str, chunk_size: int, chunk_overlap: int
) -> None:
    """Runs single file ingestion step-by-step while updating the progress bar."""
    dpg.set_value("progress_bar", 0.0)
    dpg.set_value("status_text", "Preparing document ingestion pipeline...")

    file_path = Path(file_path_str)
    if not file_path.exists():
        dpg.set_value("status_text", f"Error: File does not exist: {file_path}")
        return

    # Apply configuration overrides
    settings.DEFAULT_CHUNK_SIZE = chunk_size
    settings.DEFAULT_CHUNK_OVERLAP = chunk_overlap

    manager = WorkflowManager(
        chunker_name="hybrid_chunker",
        embedding_name="sentence_transformers",
        vector_store_name=db_backend,
    )

    try:
        # Step 1: Parsing
        dpg.set_value("progress_bar", 0.1)
        dpg.set_value("status_text", f"Parsing document: {file_path.name}...")
        start_time = time.time()

        extracted_doc = await manager.parser_router.route_and_parse(file_path)

        # Step 2: Chunking
        dpg.set_value("progress_bar", 0.35)
        dpg.set_value(
            "status_text", "Chunking document structure using Hybrid Chunker..."
        )
        chunker = plugin_manager.get_chunker(manager.chunker_name)
        if not chunker:
            raise ValueError(f"Chunker '{manager.chunker_name}' is not registered.")
        chunks = await chunker.chunk(extracted_doc)

        # Step 3: Local JSON persistence
        dpg.set_value("progress_bar", 0.5)
        dpg.set_value(
            "status_text", "Persisting structured chunks locally as JSON files..."
        )
        doc_storage_dir = settings.JSON_STORAGE_DIR / file_path.stem
        doc_storage_dir.mkdir(parents=True, exist_ok=True)
        for idx, chunk in enumerate(chunks):
            chunk_file = doc_storage_dir / f"chunk_{idx + 1}.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(chunk.model_dump(), f, indent=2, ensure_ascii=False)

        if not chunks:
            dpg.set_value("progress_bar", 1.0)
            dpg.set_value("status_text", "Ingestion completed (0 chunks generated).")
            return

        # Step 4: Embeddings
        dpg.set_value("progress_bar", 0.65)
        dpg.set_value(
            "status_text",
            f"Generating embeddings for {len(chunks)} chunks using SentenceTransformers...",
        )
        embedding_model = plugin_manager.get_embedding_model(manager.embedding_name)
        if not embedding_model:
            raise ValueError(
                f"Embedding model '{manager.embedding_name}' is not registered."
            )
        contents = [chunk.content for chunk in chunks]
        embeddings = await embedding_model.embed(contents)
        for chunk in chunks:
            chunk.embedding_status = True

        # Step 5: Indexing in Vector Database
        dpg.set_value("progress_bar", 0.85)
        dpg.set_value("status_text", f"Indexing chunks into {db_backend}...")
        vector_store = plugin_manager.get_vector_store(manager.vector_store_name)
        if not vector_store:
            raise ValueError(
                f"Vector store '{manager.vector_store_name}' is not registered."
            )
        await vector_store.add_chunks(chunks, embeddings)

        duration = time.time() - start_time
        dpg.set_value("progress_bar", 1.0)
        dpg.set_value(
            "status_text",
            f"Ingested {file_path.name} successfully in {duration:.2f} seconds!",
        )

        # Update stats UI
        dpg.set_value("stat_last_file", f"Last Ingested: {file_path.name}")
        dpg.set_value("stat_last_chunks", f"Last Chunk Count: {len(chunks)}")
        dpg.set_value("stat_last_duration", f"Last Ingestion Duration: {duration:.2f}s")
        update_db_stats()

    except Exception as e:
        logger.exception("Ingestion failed", error=str(e))
        dpg.set_value("status_text", f"Error during ingestion: {e!s}")
        dpg.set_value("progress_bar", 0.0)


async def run_dir_ingest(
    dir_path_str: str, db_backend: str, chunk_size: int, chunk_overlap: int
) -> None:
    """Runs batch directory ingestion and updates progress file-by-file."""
    dpg.set_value("progress_bar", 0.0)
    dpg.set_value("status_text", f"Scanning directory: {dir_path_str}...")

    dir_path = Path(dir_path_str)
    if not dir_path.exists():
        dpg.set_value("status_text", f"Error: Directory does not exist: {dir_path}")
        return

    # Apply configuration overrides
    settings.DEFAULT_CHUNK_SIZE = chunk_size
    settings.DEFAULT_CHUNK_OVERLAP = chunk_overlap

    manager = WorkflowManager(
        chunker_name="hybrid_chunker",
        embedding_name="sentence_transformers",
        vector_store_name=db_backend,
    )

    try:
        start_time = time.time()
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

        files = sorted(
            {fp for fp in files if fp.is_file() and not fp.name.startswith(".")}
        )
        if not files:
            dpg.set_value("status_text", "No supported documents found in directory.")
            dpg.set_value("progress_bar", 1.0)
            return

        total_files = len(files)
        total_chunks = 0

        for idx, fp in enumerate(files):
            progress = idx / total_files
            dpg.set_value("progress_bar", progress)
            dpg.set_value(
                "status_text", f"Ingesting [{idx + 1}/{total_files}]: {fp.name}..."
            )

            chunks = await manager.process_file(fp)
            total_chunks += len(chunks)

        duration = time.time() - start_time
        dpg.set_value("progress_bar", 1.0)
        dpg.set_value(
            "status_text",
            f"Ingested {total_files} documents successfully in {duration:.2f} seconds!",
        )

        # Update stats UI
        dpg.set_value("stat_last_file", f"Last Ingested: Directory '{dir_path.name}'")
        dpg.set_value("stat_last_chunks", f"Last Chunk Count: {total_chunks}")
        dpg.set_value("stat_last_duration", f"Last Ingestion Duration: {duration:.2f}s")
        update_db_stats()

    except Exception as e:
        logger.exception("Directory ingestion failed", error=str(e))
        dpg.set_value("status_text", f"Error during directory ingestion: {e!s}")
        dpg.set_value("progress_bar", 0.0)


async def run_query(query_str: str, db_backend: str) -> None:
    """Executes similarity search on the vector DB and displays output."""
    dpg.set_value("status_text", f"Executing similarity search on {db_backend}...")

    vector_store = plugin_manager.get_vector_store(db_backend)
    if not vector_store:
        dpg.set_value(
            "status_text", f"Error: Vector store '{db_backend}' is not registered."
        )
        return

    try:
        start_time = time.time()
        results = await vector_store.similarity_search(query_str, limit=5)
        duration = time.time() - start_time

        dpg.set_value("status_text", f"Search completed in {duration:.2f} seconds.")

        if not results:
            dpg.set_value(
                "search_results_output", "No matching chunks found in the database."
            )
        else:
            output_lines = []
            for idx, res in enumerate(results):
                source_info = f"{res.document_name}"
                if res.page:
                    source_info += f" | Page {res.page}"
                if res.section:
                    source_info += f" | Section: {res.section}"
                if res.rfq_number:
                    source_info += f" | RFQ: {res.rfq_number}"
                if res.customer:
                    source_info += f" | Customer: {res.customer}"

                output_lines.append(
                    f"[Result {idx + 1}] Source: {source_info} (ID: {res.chunk_id})"
                )
                output_lines.append("=" * 70)
                output_lines.append(res.content)
                output_lines.append("=" * 70 + "\n")

            dpg.set_value("search_results_output", "\n".join(output_lines))

    except Exception as e:
        logger.exception("Similarity search failed", error=str(e))
        dpg.set_value("status_text", f"Search error: {e!s}")


def update_db_stats() -> None:
    """Reads vector database chunk count and refreshes stats in the UI."""
    chroma_count = get_store_chunk_count("chromadb")
    faiss_count = get_store_chunk_count("faiss")
    dpg.set_value("stat_chroma_total", f"Total ChromaDB Chunks: {chroma_count}")
    dpg.set_value("stat_faiss_total", f"Total FAISS Chunks: {faiss_count}")


# -------------------------------------------------------------
# Button Callbacks
# -------------------------------------------------------------
_event_loop: asyncio.AbstractEventLoop | None = None
_background_tasks: set[asyncio.Task[Any]] = set()


def on_ingest_file_click() -> None:
    fp = dpg.get_value("file_path_input")
    db = dpg.get_value("db_backend")
    c_size = dpg.get_value("chunk_size")
    c_overlap = dpg.get_value("chunk_overlap")
    if not fp:
        dpg.set_value("status_text", "Error: Please select a file first.")
        return
    if _event_loop and _event_loop.is_running():
        task = _event_loop.create_task(run_ingest(fp, db, c_size, c_overlap))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    else:
        logger.error("No running event loop to schedule file ingestion")


def on_ingest_dir_click() -> None:
    dp = dpg.get_value("dir_path_input")
    db = dpg.get_value("db_backend")
    c_size = dpg.get_value("chunk_size")
    c_overlap = dpg.get_value("chunk_overlap")
    if not dp:
        dpg.set_value("status_text", "Error: Please select a directory first.")
        return
    if _event_loop and _event_loop.is_running():
        task = _event_loop.create_task(run_dir_ingest(dp, db, c_size, c_overlap))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    else:
        logger.error("No running event loop to schedule directory ingestion")


def on_search_click() -> None:
    query = dpg.get_value("search_query_input")
    db = dpg.get_value("db_backend")
    if not query:
        dpg.set_value("status_text", "Error: Please type a search query.")
        return
    if _event_loop and _event_loop.is_running():
        task = _event_loop.create_task(run_query(query, db))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    else:
        logger.error("No running event loop to schedule similarity search")


# File / Folder Picker Callbacks
def select_file_dialog_cb(sender: Any, app_data: Any) -> None:
    file_path = app_data.get("file_path_name", "")
    dpg.set_value("file_path_input", file_path)


def select_dir_dialog_cb(sender: Any, app_data: Any) -> None:
    dir_path = app_data.get("file_path_name", "")
    dpg.set_value("dir_path_input", dir_path)


# -------------------------------------------------------------
# GUI Construction and Event Loop
# -------------------------------------------------------------
async def main_gui(auto_file: str | None = None) -> None:
    # 1. Initialize Logging & Registers
    configure_logging()
    register_all_plugins()

    global _event_loop
    _event_loop = asyncio.get_running_loop()

    # 2. Setup Dear PyGui context
    dpg.create_context()
    dpg.create_viewport(
        title="KChunker Dashboard - Intelligent Chunking Controller",
        width=1000,
        height=750,
        resizable=True,
    )
    dpg.setup_dearpygui()

    # 3. Themes and Styling (Premium Slate/Cyan Color Palette)
    with dpg.theme() as global_theme, dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (15, 23, 42))  # slate-900
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (30, 41, 59))  # slate-800
        dpg.add_theme_color(dpg.mvThemeCol_Border, (51, 65, 85))  # slate-700
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (15, 23, 42))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (51, 65, 85))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (71, 85, 105))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (30, 41, 59))
        dpg.add_theme_color(dpg.mvThemeCol_Button, (14, 116, 144))  # cyan-700
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (6, 182, 212))  # cyan-500
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (34, 211, 238))  # cyan-400
        dpg.add_theme_color(dpg.mvThemeCol_Header, (30, 41, 59))
        dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (51, 65, 85))
        dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (71, 85, 105))

        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 5)
        dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8.0, 10.0)

    dpg.bind_theme(global_theme)

    # 4. File / Directory Picker Dialogs
    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=select_file_dialog_cb,
        tag="file_dialog_picker",
        width=550,
        height=400,
    ):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".pdf")
        dpg.add_file_extension(".xlsx")
        dpg.add_file_extension(".docx")
        dpg.add_file_extension(".eml")
        dpg.add_file_extension(".txt")
        dpg.add_file_extension(".md")

    with dpg.file_dialog(
        directory_selector=True,
        show=False,
        callback=select_dir_dialog_cb,
        tag="dir_dialog_picker",
        width=550,
        height=400,
    ):
        pass

    # 5. Build Primary Window layout
    with dpg.window(tag="PrimaryWindow"):
        dpg.add_spacer(height=5)
        # Banner Header
        dpg.add_text("KCHUNKER MANAGEMENT DASHBOARD", color=(34, 211, 238))
        dpg.add_text(
            "Terminal-first layout-aware document chunking and indexing engine",
            color=(148, 163, 184),
        )
        dpg.add_separator()
        dpg.add_spacer(height=5)

        # Columns container
        with dpg.group(horizontal=True):
            # Left Column (Ingestion Configuration and Triggers)
            with dpg.child_window(width=470, height=520, border=True):
                dpg.add_text("INGESTION CONTROLLER", color=(251, 146, 60))
                dpg.add_spacer(height=5)

                # Database Choice
                dpg.add_combo(
                    ["chromadb", "faiss"],
                    default_value="chromadb",
                    label="Vector Store",
                    tag="db_backend",
                    width=200,
                )

                # Param Sliders
                dpg.add_slider_int(
                    default_value=500,
                    min_value=100,
                    max_value=2000,
                    label="Chunk Size",
                    tag="chunk_size",
                    width=200,
                )
                dpg.add_slider_int(
                    default_value=50,
                    min_value=0,
                    max_value=500,
                    label="Chunk Overlap",
                    tag="chunk_overlap",
                    width=200,
                )
                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_spacer(height=10)

                # Single File Selection
                dpg.add_text("Document Ingestion:")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        hint="Select a document file...",
                        tag="file_path_input",
                        width=300,
                    )
                    dpg.add_button(
                        label="Browse",
                        callback=lambda: dpg.show_item("file_dialog_picker"),
                    )
                dpg.add_spacer(height=2)
                dpg.add_button(
                    label="Run Document Ingest",
                    callback=on_ingest_file_click,
                    width=200,
                    height=30,
                )
                dpg.add_spacer(height=15)

                # Directory Selection
                dpg.add_text("Directory Ingestion (Concurrently Processes Files):")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        hint="Select a folder...", tag="dir_path_input", width=300
                    )
                    dpg.add_button(
                        label="Browse",
                        callback=lambda: dpg.show_item("dir_dialog_picker"),
                    )
                dpg.add_spacer(height=2)
                dpg.add_button(
                    label="Run Directory Ingest",
                    callback=on_ingest_dir_click,
                    width=200,
                    height=30,
                )

            dpg.add_spacer(width=10)

            # Right Column (Search query and live metrics)
            with dpg.group():
                # Statistics child block
                with dpg.child_window(width=480, height=180, border=True):
                    dpg.add_text("LIVE STATISTICS", color=(251, 146, 60))
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        with dpg.group(width=220):
                            dpg.add_text(
                                "Total ChromaDB Chunks: --",
                                tag="stat_chroma_total",
                                color=(34, 211, 238),
                            )
                            dpg.add_text(
                                "Total FAISS Chunks: --",
                                tag="stat_faiss_total",
                                color=(34, 211, 238),
                            )
                        with dpg.group():
                            dpg.add_text("Last Ingested: N/A", tag="stat_last_file")
                            dpg.add_text("Last Chunk Count: 0", tag="stat_last_chunks")
                            dpg.add_text(
                                "Last Ingestion Duration: 0.0s",
                                tag="stat_last_duration",
                            )

                dpg.add_spacer(height=10)

                # Search query child block
                with dpg.child_window(width=480, height=330, border=True):
                    dpg.add_text("SIMILARITY SEARCH", color=(251, 146, 60))
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(
                            hint="Enter search terms here...",
                            tag="search_query_input",
                            width=330,
                        )
                        dpg.add_button(
                            label="Query DB", callback=on_search_click, width=100
                        )
                    dpg.add_spacer(height=5)
                    dpg.add_text("Search Results:")
                    dpg.add_input_text(
                        multiline=True,
                        readonly=True,
                        tag="search_results_output",
                        width=460,
                        height=210,
                    )

        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=5)

        # Bottom Bar for Progress and Status Messages
        dpg.add_progress_bar(tag="progress_bar", default_value=0.0, width=-1, height=18)
        dpg.add_spacer(height=2)
        dpg.add_text("System Ready.", tag="status_text", color=(150, 150, 150))

    dpg.set_primary_window("PrimaryWindow", True)
    dpg.show_viewport()

    # Initial stats fetch
    update_db_stats()

    # If auto-file is provided, populate it and run ingestion immediately
    if auto_file:
        dpg.set_value("file_path_input", auto_file)
        on_ingest_file_click()

    # 6. Custom Async Render Loop
    try:
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            await asyncio.sleep(0.01)
    finally:
        dpg.destroy_context()
        plugin_manager.shutdown()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="KChunker GUI Dashboard")
    parser.add_argument(
        "--file", "-f", type=str, help="Optional file path to auto-populate and ingest"
    )
    args, _ = parser.parse_known_args()

    asyncio.run(main_gui(auto_file=args.file))


if __name__ == "__main__":
    main()
