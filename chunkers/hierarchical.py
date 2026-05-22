from chunkers.semantic import SemanticChunker
from plugin_system.base import Chunk


class HierarchicalChunker:
    """Generates nested parent-child chunk structures to preserve contextual hierarchy."""

    def __init__(self, parent_size: int = 1500, child_size: int = 300) -> None:
        self.parent_size = parent_size
        self.child_size = child_size

    async def chunk_hierarchically(
        self,
        text: str,
        doc_name: str,
        doc_type: str,
        page_num: int,
        block_type: str,
        start_counter: int = 0,
    ) -> tuple[list[Chunk], int]:
        """Splits a body of text into parent and child chunks.

        Returns:
            List of generated Chunks (parents and children) and the updated chunk counter.
        """
        parent_splitter = SemanticChunker(target_chunk_size=self.parent_size)
        child_splitter = SemanticChunker(target_chunk_size=self.child_size)

        parent_texts = await parent_splitter.split_text(text)

        chunks: list[Chunk] = []
        chunk_idx = start_counter

        for _p_idx, p_text in enumerate(parent_texts):
            parent_id = f"{doc_name}_p{page_num}_parent_{chunk_idx}"
            chunk_idx += 1

            # Split the parent text into smaller child texts
            child_texts = await child_splitter.split_text(p_text)
            child_ids = []
            child_chunks: list[Chunk] = []

            for c_idx, c_text in enumerate(child_texts):
                child_id = f"{doc_name}_p{page_num}_child_{parent_id}_{c_idx}"
                child_ids.append(child_id)

                child_chunks.append(
                    Chunk(
                        chunk_id=child_id,
                        document_name=doc_name,
                        document_type=doc_type,
                        chunk_type="CHILD",
                        page=page_num,
                        section=block_type,
                        content=c_text,
                        related_chunks=[parent_id],
                        metadata={"is_child": True, "parent_id": parent_id},
                    )
                )

            # Create parent chunk
            parent_chunk = Chunk(
                chunk_id=parent_id,
                document_name=doc_name,
                document_type=doc_type,
                chunk_type="PARENT",
                page=page_num,
                section=block_type,
                content=p_text,
                related_chunks=child_ids,
                metadata={"is_parent": True, "child_ids": child_ids},
            )

            chunks.append(parent_chunk)
            chunks.extend(child_chunks)

        return chunks, chunk_idx
