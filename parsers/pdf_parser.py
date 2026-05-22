from pathlib import Path
from typing import Any

from plugin_system.base import BaseParser, ExtractedDocument
from utils.logging import logger

fitz: Any = None
try:
    import fitz as fitz_module  # PyMuPDF

    fitz = fitz_module
except ImportError:
    pass

pdfplumber: Any = None
try:
    import pdfplumber as pdfplumber_module

    pdfplumber = pdfplumber_module
except ImportError:
    pass


class PDFParser(BaseParser):
    """Parses native PDF documents, extracting text and tables page-by-page."""

    @property
    def name(self) -> str:
        return "pdf_parser"

    @property
    def version(self) -> str:
        return "1.0.0"

    def can_parse(self, file_path: Path, mime_type: str) -> bool:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf" or "pdf" in mime_type:
            from parsers.router import ParserRouter

            return ParserRouter.detect_document_type(file_path) == "PDF_NATIVE"
        return False

    async def parse(self, file_path: Path) -> ExtractedDocument:
        logger.info("Parsing PDF document", path=str(file_path))

        raw_text_parts: list[str] = []
        pages_metadata: list[dict[str, Any]] = []
        extracted_tables: list[dict[str, Any]] = []

        # 1. Extract Text using PyMuPDF (fitz) if available
        if fitz:
            try:
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    raw_text_parts.append(page_text)

                    pages_metadata.append(
                        {
                            "page_num": page_num + 1,
                            "text": page_text,
                            "width": page.rect.width,
                            "height": page.rect.height,
                        }
                    )
                doc.close()
            except Exception as e:
                logger.error("Failed to parse PDF text via PyMuPDF", error=str(e))
        else:
            logger.warn(
                "PyMuPDF is not installed; skipping PyMuPDF fast text extraction"
            )

        # 2. Extract Tables using pdfplumber if available
        if pdfplumber:
            try:
                with pdfplumber.open(file_path) as pdf:
                    # If PyMuPDF wasn't used/failed, use pdfplumber for text fallback
                    fallback_text = not raw_text_parts

                    for idx, page in enumerate(pdf.pages):
                        page_num = idx + 1

                        if fallback_text:
                            page_text = page.extract_text() or ""
                            raw_text_parts.append(page_text)
                            pages_metadata.append(
                                {
                                    "page_num": page_num,
                                    "text": page_text,
                                    "width": page.width,
                                    "height": page.height,
                                }
                            )

                        # Extract tables
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables):
                            if table:
                                # Simple normalization of table cells
                                cleaned_table = [
                                    [str(cell or "").strip() for cell in row]
                                    for row in table
                                ]
                                extracted_tables.append(
                                    {
                                        "page": page_num,
                                        "table_index": table_idx,
                                        "data": cleaned_table,
                                    }
                                )
            except Exception as e:
                logger.error("Failed to extract tables via pdfplumber", error=str(e))
        else:
            logger.warn(
                "pdfplumber is not installed; skipping table extraction fallback"
            )

        combined_text = "\n--- PAGE BREAK ---\n".join(raw_text_parts)

        return ExtractedDocument(
            document_name=file_path.name,
            document_path=str(file_path),
            document_type="PDF",
            raw_text=combined_text,
            pages=pages_metadata,
            tables=extracted_tables,
            emails=[],
            metadata={"num_pages": len(pages_metadata)},
        )
