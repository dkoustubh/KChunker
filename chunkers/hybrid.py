import re

from chunkers.email_chunker import EmailChunker
from chunkers.hierarchical import HierarchicalChunker
from chunkers.semantic import SemanticChunker
from chunkers.table_aware import TableAwareChunker
from configs.config import settings
from plugin_system.base import BaseChunker, Chunk, ExtractedDocument
from utils.logging import logger


class HybridChunker(BaseChunker):
    """Production hybrid chunker combining semantic, hierarchical, table, email, and layout chunking."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        similarity_threshold: float = 0.7,
    ) -> None:
        self.chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP
        self.similarity_threshold = similarity_threshold

    @property
    def name(self) -> str:
        return "hybrid_chunker"

    @property
    def version(self) -> str:
        return "1.0.0"

    async def chunk(self, doc: ExtractedDocument) -> list[Chunk]:
        logger.info(
            "Starting hybrid chunking orchestration",
            doc_name=doc.document_name,
            doc_type=doc.document_type,
        )

        chunks: list[Chunk] = []
        chunk_counter = 0

        # Extract document-level metadata (like RFQ or Client)
        rfq_info = self._extract_rfq_metadata(doc.raw_text)
        customer = rfq_info.get("customer")
        rfq_number = rfq_info.get("rfq_number")
        revision = rfq_info.get("revision")

        # 1. Handle Email chains
        if doc.document_type == "EMAIL" and doc.emails:
            email_chunker = EmailChunker()
            email_chunks, chunk_counter = email_chunker.chunk_emails(
                doc.emails, doc.document_name, doc.document_type, chunk_counter
            )
            chunks.extend(email_chunks)

        # 2. Handle Spreadsheets (Excel)
        elif doc.document_type == "EXCEL":
            table_chunker = TableAwareChunker()
            for table_dict in doc.tables:
                table_chunks, chunk_counter = table_chunker.chunk_table(
                    table_dict, doc.document_name, doc.document_type, chunk_counter
                )
                chunks.extend(table_chunks)

        # 3. Handle PDF, DOCX, TXT, IMAGE with potential layout and tables
        else:
            # Process any detected tables first
            table_chunker = TableAwareChunker()
            for table_dict in doc.tables:
                table_chunks, chunk_counter = table_chunker.chunk_table(
                    table_dict, doc.document_name, doc.document_type, chunk_counter
                )
                chunks.extend(table_chunks)

            # Process layout/page text
            hierarchical_chunker = HierarchicalChunker(
                parent_size=self.chunk_size * 3, child_size=self.chunk_size
            )
            semantic_chunker = SemanticChunker(target_chunk_size=self.chunk_size)

            for page in doc.pages:
                page_num = page.get("page_num", 1)

                # Check if layout blocks are available
                if page.get("blocks"):
                    for block in page["blocks"]:
                        # Block can be dict-like (e.g. from OCR or LayoutDetector)
                        content = block.get("content", "")
                        coords = block.get("coordinates", [])
                        block_type = block.get("block_type", "TEXT")

                        if not content.strip():
                            continue

                        # Chunk the block text using hierarchical or semantic chunker
                        if len(content) > self.chunk_size * 2:
                            # Use parent-child splitting for large blocks
                            block_chunks, chunk_counter = (
                                await hierarchical_chunker.chunk_hierarchically(
                                    content,
                                    doc.document_name,
                                    doc.document_type,
                                    page_num,
                                    block_type,
                                    chunk_counter,
                                )
                            )
                        else:
                            # Use semantic/recursive splitting for smaller blocks
                            split_texts = await semantic_chunker.split_text(content)
                            block_chunks = []
                            for s_idx, st in enumerate(split_texts):
                                block_chunks.append(
                                    Chunk(
                                        chunk_id=f"{doc.document_name}_p{page_num}_b_{chunk_counter}",
                                        document_name=doc.document_name,
                                        document_type=doc.document_type,
                                        chunk_type=block_type,
                                        page=page_num,
                                        section=block_type,
                                        content=st,
                                        metadata={"block_part": s_idx},
                                    )
                                )
                                chunk_counter += 1

                        # Ingest coordinates into the chunk models
                        for bc in block_chunks:
                            bc.coordinates = coords
                            chunks.append(bc)
                else:
                    # Parse page raw text directly if no blocks
                    page_text = page.get("text", "")
                    if not page_text.strip():
                        continue

                    split_texts = await semantic_chunker.split_text(page_text)
                    for st in split_texts:
                        chunks.append(
                            Chunk(
                                chunk_id=f"{doc.document_name}_p{page_num}_{chunk_counter}",
                                document_name=doc.document_name,
                                document_type=doc.document_type,
                                chunk_type="TEXT",
                                page=page_num,
                                content=st,
                            )
                        )
                        chunk_counter += 1

        # Enrich metadata for all chunks
        for chunk in chunks:
            if customer:
                chunk.customer = customer
            if rfq_number:
                chunk.rfq_number = rfq_number
            if revision:
                chunk.revision = revision

            # Add general metadata
            chunk.metadata.update(
                {
                    "doc_name": doc.document_name,
                    "doc_type": doc.document_type,
                }
            )

        logger.info("Hybrid chunking complete", total_chunks=len(chunks))
        return chunks

    def _extract_rfq_metadata(self, text: str) -> dict[str, str]:
        """Helper to extract common RFQ fields (customer, RFQ number, revision) from raw text."""
        metadata = {}

        # Look for RFQ number
        rfq_patterns = [
            r"RFQ\s*(?:No|Number|Ref)?\s*[:#-]?\s*([A-Za-z0-9-_]+)",
            r"Request\s*For\s*Quote\s*(?:No|Number|Ref)?\s*[:#-]?\s*([A-Za-z0-9-_]+)",
            r"Inquiry\s*(?:No|Number)?\s*[:#-]?\s*([A-Za-z0-9-_]+)",
        ]
        for pattern in rfq_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["rfq_number"] = match.group(1).strip()
                break

        # Look for revision
        rev_patterns = [
            r"\bRevision\s*(?:No|Number|Level)?\s*[:#-]?\s*([A-Za-z0-9.]+)",
            r"\bRev\s*(?:No|Number|Level)?\s*[:#-]?\s*([A-Za-z0-9.]+)",
        ]
        for pattern in rev_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["revision"] = match.group(1).strip()
                break

        # Look for customer/client name
        client_patterns = [
            r"Client\s*:\s*([^\n]+)",
            r"Customer\s*:\s*([^\n]+)",
            r"Issued\s*To\s*:\s*([^\n]+)",
        ]
        for pattern in client_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Clean up match
                val = match.group(1).strip()
                if len(val) < 50:
                    metadata["customer"] = val
                    break

        return metadata
