import mimetypes
from pathlib import Path

from plugin_system.base import ExtractedDocument
from plugin_system.manager import plugin_manager
from utils.logging import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class ParserRouter:
    """Classifies files and routes them to matching parser plugins."""

    @staticmethod
    def detect_document_type(file_path: Path) -> str:
        """Determines the document classification based on type and content.

        Classifications:
        - PDF_NATIVE
        - PDF_SCANNED
        - EXCEL
        - DOCX
        - EMAIL
        - TXT
        - IMAGE
        - UNKNOWN
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return ParserRouter._classify_pdf(file_path)
        elif suffix in [".xlsx", ".xls", ".csv"]:
            return "EXCEL"
        elif suffix in [".docx", ".doc"]:
            return "DOCX"
        elif suffix in [".eml", ".msg"]:
            return "EMAIL"
        elif suffix in [".txt", ".md", ".json"]:
            return "TXT"
        elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return "IMAGE"

        # Fallback to MIME type detection
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if "pdf" in mime_type:
                return ParserRouter._classify_pdf(file_path)
            elif "sheet" in mime_type or "excel" in mime_type or "csv" in mime_type:
                return "EXCEL"
            elif "word" in mime_type:
                return "DOCX"
            elif "message" in mime_type or "rfc822" in mime_type:
                return "EMAIL"
            elif "text" in mime_type:
                return "TXT"
            elif "image" in mime_type:
                return "IMAGE"

        return "UNKNOWN"

    @staticmethod
    def _classify_pdf(file_path: Path) -> str:
        """Determines if a PDF is native or scanned by analyzing text density on page 1-3."""
        if not fitz:
            logger.warn(
                "PyMuPDF (fitz) is not installed. "
                "Defaulting PDF classification to PDF_NATIVE."
            )
            return "PDF_NATIVE"

        try:
            doc = fitz.open(file_path)
            # Sample up to first 3 pages
            sample_pages = min(len(doc), 3)
            if sample_pages == 0:
                return "PDF_SCANNED"

            total_text_length = 0
            for i in range(sample_pages):
                page = doc.load_page(i)
                total_text_length += len(page.get_text().strip())

            # Less than 50 chars average -> scanned/image-heavy
            avg_text_length = total_text_length / sample_pages
            if avg_text_length < 50:
                logger.info(
                    "PDF classified as scanned / image-heavy due to low text density",
                    path=str(file_path),
                    avg_chars=avg_text_length,
                )
                return "PDF_SCANNED"

            logger.info(
                "PDF classified as native due to text presence",
                path=str(file_path),
                avg_chars=avg_text_length,
            )
            return "PDF_NATIVE"

        except Exception as e:
            logger.error(
                "Error classifying PDF, defaulting to native",
                path=str(file_path),
                error=str(e),
            )
            return "PDF_NATIVE"

    async def route_and_parse(self, file_path: Path) -> ExtractedDocument:
        """Classifies the file type, locates parser plugin, and parses the document."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_type = self.detect_document_type(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"

        logger.info(
            "Routing document for parsing", path=str(file_path), detected_type=doc_type
        )

        parsers = plugin_manager.get_all_parsers()

        # 1. Search for a parser plugin that explicitly can parse this document
        for parser in parsers:
            if parser.can_parse(file_path, mime_type):
                logger.info(
                    "Parser match found", parser_name=parser.name, path=str(file_path)
                )
                return await parser.parse(file_path)

        # 2. If no custom plugin, raise or fallback to basic parser if available
        raise ValueError(
            f"No suitable parser registered to handle file type: "
            f"{doc_type} (MIME: {mime_type})"
        )
