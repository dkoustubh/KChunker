import os
import tempfile
from pathlib import Path
from typing import Any

from ocr.layout_detection import LayoutDetector
from ocr.paddle_ocr import PaddleOCRWrapper
from plugin_system.base import BaseParser, ExtractedDocument
from utils.logging import logger

fitz: Any = None
try:
    import fitz as fitz_module  # PyMuPDF

    fitz = fitz_module
except ImportError:
    pass


class OCRParser(BaseParser):
    """Parses scanned PDFs and images using PaddleOCR, extracting spatial blocks."""

    def __init__(self, ocr_engine: PaddleOCRWrapper | None = None) -> None:
        self.ocr_engine = ocr_engine or PaddleOCRWrapper()

    @property
    def name(self) -> str:
        return "ocr_parser"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> None:
        self.ocr_engine.initialize()

    def can_parse(self, file_path: Path, mime_type: str) -> bool:
        suffix = file_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"] or "image" in mime_type:
            return True
        if suffix == ".pdf" or "pdf" in mime_type:
            # Let's import ParserRouter locally to avoid circular dependency
            from parsers.router import ParserRouter

            return ParserRouter.detect_document_type(file_path) == "PDF_SCANNED"
        return False

    async def parse(self, file_path: Path) -> ExtractedDocument:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return await self._parse_scanned_pdf(file_path)
        else:
            return await self._parse_image(file_path)

    async def _parse_scanned_pdf(self, file_path: Path) -> ExtractedDocument:
        logger.info("Parsing scanned PDF via OCR", path=str(file_path))
        if not fitz:
            raise ImportError(
                "PyMuPDF (fitz) is required to render scanned PDF pages for OCR."
            )

        raw_text_parts: list[str] = []
        pages_metadata: list[dict[str, Any]] = []

        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                logger.info(
                    "Running OCR on PDF page",
                    page_num=page_num + 1,
                    total_pages=len(doc),
                )

                # Render page to a temporary image
                pix = page.get_pixmap(dpi=150)
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as temp_img:
                    temp_img_path = Path(temp_img.name)

                try:
                    pix.save(str(temp_img_path))

                    # Extract OCR regions
                    regions = await self.ocr_engine.extract_regions(temp_img_path)

                    # Group regions into layout blocks
                    layout_blocks = LayoutDetector.group_ocr_regions(
                        regions, page_num + 1
                    )

                    # Combine text for the page
                    page_text = "\n\n".join([b.content for b in layout_blocks])
                    raw_text_parts.append(page_text)

                    # Store page metadata with layout blocks
                    pages_metadata.append(
                        {
                            "page_num": page_num + 1,
                            "text": page_text,
                            "width": page.rect.width,
                            "height": page.rect.height,
                            "blocks": [b.model_dump() for b in layout_blocks],
                        }
                    )
                finally:
                    # Clean up temp image
                    if temp_img_path.exists():
                        os.unlink(temp_img_path)

            doc.close()
        except Exception as e:
            logger.error(
                "Failed to parse scanned PDF with OCR",
                path=str(file_path),
                error=str(e),
            )
            raise e

        combined_text = "\n--- PAGE BREAK ---\n".join(raw_text_parts)

        return ExtractedDocument(
            document_name=file_path.name,
            document_path=str(file_path),
            document_type="PDF",
            raw_text=combined_text,
            pages=pages_metadata,
            tables=[],
            emails=[],
            metadata={"num_pages": len(pages_metadata), "is_scanned": True},
        )

    async def _parse_image(self, file_path: Path) -> ExtractedDocument:
        logger.info("Parsing image via OCR", path=str(file_path))

        try:
            regions = await self.ocr_engine.extract_regions(file_path)
            layout_blocks = LayoutDetector.group_ocr_regions(regions, 1)

            image_text = "\n\n".join([b.content for b in layout_blocks])

            pages_metadata = [
                {
                    "page_num": 1,
                    "text": image_text,
                    "blocks": [b.model_dump() for b in layout_blocks],
                }
            ]
        except Exception as e:
            logger.error(
                "Failed to run OCR on image", path=str(file_path), error=str(e)
            )
            raise e

        return ExtractedDocument(
            document_name=file_path.name,
            document_path=str(file_path),
            document_type="IMAGE",
            raw_text=image_text,
            pages=pages_metadata,
            tables=[],
            emails=[],
            metadata={"is_image": True},
        )
