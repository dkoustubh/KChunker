import pytest

from chunkers.email_chunker import EmailChunker
from chunkers.hierarchical import HierarchicalChunker
from chunkers.semantic import SemanticChunker
from chunkers.table_aware import TableAwareChunker


@pytest.mark.asyncio
async def test_semantic_chunker_fallback() -> None:
    # Test semantic chunker fallback when no embedding model is configured
    chunker = SemanticChunker(target_chunk_size=100)
    text = (
        "This is a sentence. And here is another sentence. "
        "This should fall back to recursive character splitting."
    )
    chunks = await chunker.split_text(text)

    assert len(chunks) > 0
    # Every chunk should be within the size limit (approx)
    for c in chunks:
        assert len(c) <= 100


@pytest.mark.asyncio
async def test_hierarchical_chunker() -> None:
    chunker = HierarchicalChunker(parent_size=200, child_size=50)
    text = (
        "This is parent context paragraph. It is long and has sentences. "
        "We want to split this block hierarchically so that we have smaller "
        "child chunks and a larger parent chunk containing all detail. "
        "This preserves both precision and context."
    )

    chunks, _ = await chunker.chunk_hierarchically(
        text=text,
        doc_name="test_doc.txt",
        doc_type="TXT",
        page_num=1,
        block_type="TEXT",
        start_counter=0,
    )

    # We should have at least 1 parent chunk and multiple child chunks
    parent_chunks = [c for c in chunks if c.chunk_type == "PARENT"]
    child_chunks = [c for c in chunks if c.chunk_type == "CHILD"]

    assert len(parent_chunks) > 0
    assert len(child_chunks) > 0

    # Verify child chunks are linked to parent chunk
    parent_ids = {p.chunk_id for p in parent_chunks}
    for child in child_chunks:
        # Check if the child relates to a parent
        has_parent_link = False
        for rel in child.related_chunks:
            if rel in parent_ids:
                has_parent_link = True
                break
        assert has_parent_link, "Child chunk must be linked to its parent chunk ID"


def test_table_aware_chunker_small() -> None:
    chunker = TableAwareChunker()
    table_dict = {
        "table_index": 0,
        "data": [
            ["Header 1", "Header 2", "Header 3"],
            ["Val 1", "Val 2", "Val 3"],
            ["Val 4", "Val 5", "Val 6"],
        ],
    }

    chunks, _ = chunker.chunk_table(
        table_dict=table_dict, doc_name="specs.xlsx", doc_type="EXCEL", start_counter=0
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == "TABLE"
    assert "Header 1 | Header 2 | Header 3" in chunk.content
    assert "Val 1 | Val 2 | Val 3" in chunk.content
    assert "Val 4 | Val 5 | Val 6" in chunk.content


def test_table_aware_chunker_splitting() -> None:
    # Using small max_rows to trigger splitting of tables
    chunker = TableAwareChunker(max_rows=2)
    table_dict = {
        "table_index": 1,
        "data": [
            ["Col A", "Col B"],
            ["Row 1 A", "Row 1 B"],
            ["Row 2 A", "Row 2 B"],
            ["Row 3 A", "Row 3 B"],
        ],
    }

    chunks, _ = chunker.chunk_table(
        table_dict=table_dict, doc_name="specs.xlsx", doc_type="EXCEL", start_counter=0
    )

    # 3 data rows split into chunks of max 2 rows each.
    # Chunk 1: Header + Row 1 + Row 2 (2 data rows)
    # Chunk 2: Header + Row 3 (1 data row)
    assert len(chunks) == 2

    # Verify header is repeated in both chunks
    assert "Col A | Col B" in chunks[0].content
    assert "Col A | Col B" in chunks[1].content

    assert "Row 1 A" in chunks[0].content
    assert "Row 3 A" in chunks[1].content
    assert "Row 3 A" not in chunks[0].content


def test_email_chunker() -> None:
    chunker = EmailChunker()
    emails = [
        {
            "sender": "alice@company.com",
            "recipient": "bob@company.com",
            "subject": "Project X updates",
            "date": "2026-05-22 10:00:00",
            "body": "Hi Bob, can you send the drawings?",
        },
        {
            "sender": "bob@company.com",
            "recipient": "alice@company.com",
            "subject": "Re: Project X updates",
            "date": "2026-05-22 10:05:00",
            "body": "Hi Alice, yes I will send them shortly.",
        },
    ]

    chunks, _ = chunker.chunk_emails(
        emails=emails, doc_name="email_trail.eml", doc_type="EMAIL", start_counter=0
    )

    assert len(chunks) == 2
    for c in chunks:
        assert c.chunk_type == "EMAIL"
        assert c.document_name == "email_trail.eml"

    assert "alice@company.com" in chunks[0].content
    assert "bob@company.com" in chunks[1].content

    # Verify they link back to each other or have trail reference
    assert chunks[1].related_chunks[0] == chunks[0].chunk_id
