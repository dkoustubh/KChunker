from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from embeddings.sentence_transformers import SentenceTransformersModel
from plugin_system.base import Chunk
from plugin_system.manager import plugin_manager
from vector_db.chroma_db import ChromaVectorStore
from vector_db.faiss_db import FAISSVectorStore


@pytest.mark.asyncio
async def test_sentence_transformers_model_mock() -> None:
    # Initialize and check name and version
    model = SentenceTransformersModel(model_name="mock-model")
    assert model.name == "sentence_transformers"
    assert model.version == "1.0.0"

    # Mock the internal SentenceTransformer model to prevent downloading
    mock_st = MagicMock()
    # Let encode return a numpy array of dimension 3
    mock_st.encode.return_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with patch(
        "embeddings.sentence_transformers.SentenceTransformer", return_value=mock_st
    ):
        model.initialize()
        embeddings = await model.embed(["hello", "world"])
        assert embeddings == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        mock_st.encode.assert_called_once_with(
            ["hello", "world"], show_progress_bar=False
        )


@pytest.mark.asyncio
async def test_sentence_transformers_model_fallback() -> None:
    # Force ImportError on sentence_transformers import to test fallback
    with patch("embeddings.sentence_transformers.SentenceTransformer", None):
        model = SentenceTransformersModel(model_name="mock-model")
        model.initialize()
        embeddings = await model.embed(["hello"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert all(val == 0.0 for val in embeddings[0])


@pytest.mark.asyncio
async def test_faiss_vector_store(tmp_path: Any) -> None:
    # Initialize FAISS Vector Store in a temp directory
    store = FAISSVectorStore(db_dir=tmp_path)
    store.initialize()

    assert store.name == "faiss"
    assert store.version == "1.0.0"

    # Create dummy chunks and embeddings
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            document_name="doc1.txt",
            document_type="TXT",
            chunk_type="TEXT",
            page=1,
            section="Section A",
            content="This is the content of chunk one.",
            metadata={"priority": "high", "topic": "science"},
        ),
        Chunk(
            chunk_id="chunk_2",
            document_name="doc1.txt",
            document_type="TXT",
            chunk_type="TEXT",
            page=2,
            section="Section B",
            content="This is the content of chunk two.",
            metadata={"priority": "low", "topic": "history"},
        ),
    ]
    # Use 384 dimensional embeddings (default)
    embeddings = [[0.0] * 384 for _ in range(2)]
    embeddings[0][0] = 1.0  # Make them distinct
    embeddings[1][1] = 1.0

    # Add chunks
    success = await store.add_chunks(chunks, embeddings)
    assert success is True

    # Register the embedding model so similarity_search can query it
    mock_model = MagicMock()
    mock_model.embed = AsyncMock(return_value=[[1.0] + [0.0] * 383])

    # We patch plugin_manager.get_embedding_model
    with patch.object(plugin_manager, "get_embedding_model", return_value=mock_model):
        # 1. Test basic similarity search
        results = await store.similarity_search("query text", limit=2)
        assert len(results) == 2
        # First result should be chunk_1 since its embedding matches query better
        assert results[0].chunk_id == "chunk_1"

        # 2. Test search with metadata filter (matching page)
        results_filtered = await store.similarity_search(
            "query text", limit=2, filters={"page": 2}
        )
        assert len(results_filtered) == 1
        assert results_filtered[0].chunk_id == "chunk_2"

        # 3. Test search with nested metadata dict filter
        results_filtered_meta = await store.similarity_search(
            "query text", limit=2, filters={"topic": "science"}
        )
        assert len(results_filtered_meta) == 1
        assert results_filtered_meta[0].chunk_id == "chunk_1"

    # Reload store from the same directory to verify persistence
    store_reload = FAISSVectorStore(db_dir=tmp_path)
    store_reload.initialize()
    assert len(store_reload._chunks) == 2
    assert store_reload._chunks[0].chunk_id == "chunk_1"


@pytest.mark.asyncio
async def test_chroma_vector_store(tmp_path: Any) -> None:
    # Initialize Chroma Vector Store in a temp directory
    store = ChromaVectorStore(db_dir=tmp_path)
    store.initialize()

    assert store.name == "chromadb"
    assert store.version == "1.0.0"

    chunks = [
        Chunk(
            chunk_id="c1",
            document_name="specs.pdf",
            document_type="PDF",
            chunk_type="TEXT",
            page=1,
            content="Chroma chunk one content.",
            metadata={"status": "draft", "category": "engineering"},
        ),
        Chunk(
            chunk_id="c2",
            document_name="specs.pdf",
            document_type="PDF",
            chunk_type="TEXT",
            page=2,
            content="Chroma chunk two content.",
            metadata={"status": "final", "category": "finance"},
        ),
    ]
    embeddings = [[0.0] * 384 for _ in range(2)]
    embeddings[0][0] = 1.0
    embeddings[1][1] = 1.0

    # Add chunks
    success = await store.add_chunks(chunks, embeddings)
    assert success is True

    # Register mock embedding model
    mock_model = MagicMock()
    mock_model.embed = AsyncMock(return_value=[[1.0] + [0.0] * 383])

    with patch.object(plugin_manager, "get_embedding_model", return_value=mock_model):
        # 1. Search query
        results = await store.similarity_search("query text", limit=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"

        # 2. Search with filter
        results_filtered = await store.similarity_search(
            "query text", limit=2, filters={"page": 2}
        )
        assert len(results_filtered) == 1
        assert results_filtered[0].chunk_id == "c2"

        # 3. Search with status filter
        results_status = await store.similarity_search(
            "query text", limit=2, filters={"status": "draft"}
        )
        assert len(results_status) == 1
        assert results_status[0].chunk_id == "c1"
