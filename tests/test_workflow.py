import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkers.hybrid import HybridChunker
from configs.config import settings
from parsers.txt_parser import TXTParser
from plugin_system.manager import plugin_manager
from workflows.manager import WorkflowManager


@pytest.mark.asyncio
async def test_workflow_manager_directory_ingestion(tmp_path: Any) -> None:
    # 1. Setup temporary directory structure for files and JSON outputs
    input_dir = tmp_path / "documents"
    input_dir.mkdir()

    output_dir = tmp_path / "output_json"
    output_dir.mkdir()

    # Create test files
    file1 = input_dir / "doc1.txt"
    file1.write_text(
        "RFQ No: RFQ-2026-X1\n"
        "Customer: Acme Corporation\n"
        "Revision: 1.2\n"
        "This is the first main paragraph of text for doc1.\n"
        "It contains some general details about engineering requirements.",
        encoding="utf-8",
    )

    file2 = input_dir / "doc2.md"
    file2.write_text(
        "This is the first paragraph of markdown file 2.\n"
        "We also mention Inquiry No: INQ-777 here.\n"
        "And we are doing this for Client: Wayne Enterprises.",
        encoding="utf-8",
    )

    # 2. Register core plugins needed for the test
    plugin_manager.register_parser(TXTParser())
    plugin_manager.register_chunker(HybridChunker())

    # Mock SentenceTransformersModel to avoid loading it from network
    mock_model = MagicMock()
    mock_model.name = "sentence_transformers"
    mock_model.version = "1.0.0"
    mock_model.embed = AsyncMock(side_effect=lambda texts: [[0.1] * 384 for _ in texts])

    # Register the mock embedding model
    plugin_manager.register_embedding_model(mock_model)

    # Let's configure FAISS to use a temp directory
    faiss_dir = tmp_path / "faiss_db"
    from vector_db.faiss_db import FAISSVectorStore

    faiss_store = FAISSVectorStore(db_dir=faiss_dir)
    plugin_manager.register_vector_store(faiss_store)

    # Override settings for JSON storage in WorkflowManager
    settings.JSON_STORAGE_DIR = output_dir

    # 3. Initialize WorkflowManager
    manager = WorkflowManager(
        chunker_name="hybrid_chunker",
        embedding_name="sentence_transformers",
        vector_store_name="faiss",
    )

    # 4. Process directory
    chunks = await manager.process_directory(input_dir)

    # 5. Assertions
    assert len(chunks) > 0

    doc_names = {c.document_name for c in chunks}
    assert "doc1.txt" in doc_names
    assert "doc2.md" in doc_names

    # Check that metadata was extracted correctly
    doc1_chunks = [c for c in chunks if c.document_name == "doc1.txt"]
    assert len(doc1_chunks) > 0
    for c in doc1_chunks:
        assert c.rfq_number == "RFQ-2026-X1"
        assert c.customer == "Acme Corporation"
        assert c.revision == "1.2"
        assert c.embedding_status is True

    doc2_chunks = [c for c in chunks if c.document_name == "doc2.md"]
    assert len(doc2_chunks) > 0
    for c in doc2_chunks:
        assert c.rfq_number == "INQ-777"
        assert c.customer == "Wayne Enterprises."
        assert c.embedding_status is True

    # Check that JSON files were stored on disk
    json_files = list(output_dir.rglob("*.json"))
    assert len(json_files) == len(chunks)

    # Let's read one of the JSON files to verify content
    with open(json_files[0], encoding="utf-8") as f:
        data = json.load(f)
        assert "chunk_id" in data
        assert "content" in data

    # Verify that the chunks were stored in FAISS and we can query them
    search_results = await faiss_store.similarity_search("Acme", limit=5)
    assert len(search_results) > 0
