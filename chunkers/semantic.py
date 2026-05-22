import re

from plugin_system.base import BaseEmbeddingModel
from utils.logging import logger


class SemanticChunker:
    """Chunks text semantically by analyzing similarity between adjacent sentences, falling back to recursive character splitting."""

    def __init__(
        self,
        target_chunk_size: int = 500,
        similarity_threshold: float = 0.7,
        embedding_model: BaseEmbeddingModel | None = None,
    ) -> None:
        self.target_chunk_size = target_chunk_size
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

    async def split_text(self, text: str) -> list[str]:
        """Asynchronously splits text into semantic chunks."""
        if not text.strip():
            return []

        # If embedding model is available, attempt similarity-based chunking
        if self.embedding_model:
            try:
                return await self._split_semantically(text)
            except Exception as e:
                logger.warn(
                    "Semantic chunking via embedding failed; falling back to recursive split",
                    error=str(e),
                )

        return self._split_recursive(text)

    async def _split_semantically(self, text: str) -> list[str]:
        if not self.embedding_model:
            raise ValueError("Embedding model must be provided for semantic splitting.")
        # Split text into sentences (using a lookbehind to keep punctuation)
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            return sentences

        # Generate embeddings for all sentences
        embeddings = await self.embedding_model.embed(sentences)

        chunks: list[str] = []
        current_chunk_sentences = [sentences[0]]

        # Calculate cosine similarity between consecutive sentence embeddings
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])

            # Check length of the potential chunk
            current_len = (
                sum(len(s) for s in current_chunk_sentences)
                + len(current_chunk_sentences)
                - 1
            )
            next_len = len(sentences[i + 1])

            # If similarity is low and the current chunk has sufficient length, or if adding the next sentence exceeds size limit
            if (
                sim < self.similarity_threshold
                and current_len >= self.target_chunk_size // 2
            ) or (current_len + next_len > self.target_chunk_size):
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i + 1]]
            else:
                current_chunk_sentences.append(sentences[i + 1])

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def _split_recursive(self, text: str) -> list[str]:
        """Recursively splits text using separators (double newlines, single newlines, spaces) to fit target_chunk_size."""
        separators = ["\n\n", "\n", " ", ""]

        def _split(txt: str, separator_idx: int) -> list[str]:
            if len(txt) <= self.target_chunk_size:
                return [txt]

            if separator_idx >= len(separators):
                # Hard chunking if we ran out of separators
                return [
                    txt[i : i + self.target_chunk_size]
                    for i in range(0, len(txt), self.target_chunk_size)
                ]

            sep = separators[separator_idx]
            parts = txt.split(sep) if sep else list(txt)

            chunks = []
            current_chunk: list[str] = []
            current_len = 0

            for part in parts:
                part_len = len(part)
                # If a single part exceeds target size, split it recursively using next separator
                if part_len > self.target_chunk_size:
                    if current_chunk:
                        chunks.append(sep.join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    chunks.extend(_split(part, separator_idx + 1))
                elif (
                    current_len + part_len + (len(sep) if current_chunk else 0)
                    > self.target_chunk_size
                ):
                    if current_chunk:
                        chunks.append(sep.join(current_chunk))
                    current_chunk = [part]
                    current_len = part_len
                else:
                    current_chunk.append(part)
                    current_len += part_len + (
                        len(sep) if len(current_chunk) > 1 else 0
                    )

            if current_chunk:
                chunks.append(sep.join(current_chunk))

            return chunks

        return _split(text, 0)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
