from pathlib import Path

from plugin_system.base import BaseParser, ExtractedDocument
from utils.logging import logger


class TXTParser(BaseParser):
    """Parses raw text and markdown files."""

    @property
    def name(self) -> str:
        return "txt_parser"

    @property
    def version(self) -> str:
        return "1.0.0"

    def can_parse(self, file_path: Path, mime_type: str) -> bool:
        suffix = file_path.suffix.lower()
        return (
            suffix in [".txt", ".md", ".json", ".ini", ".conf"] or "text" in mime_type
        )

    async def parse(self, file_path: Path) -> ExtractedDocument:
        logger.info("Parsing Plain Text document", path=str(file_path))

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.error("Failed to read text file", path=str(file_path), error=str(e))
            raise e

        return ExtractedDocument(
            document_name=file_path.name,
            document_path=str(file_path),
            document_type="TXT",
            raw_text=content,
            pages=[{"page_num": 1, "text": content}],
            tables=[],
            emails=[],
            metadata={"character_count": len(content)},
        )
