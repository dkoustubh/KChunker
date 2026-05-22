from pathlib import Path
from typing import Any

from ocr.base import BaseOCR
from utils.logging import logger

try:
    from paddleocr import PaddleOCR as PDOCR  # noqa: N814
except ImportError:
    PDOCR = None


class PaddleOCRWrapper(BaseOCR):
    """OCR processor wrapping PaddleOCR for image text extraction."""

    def __init__(self, lang: str = "en", use_gpu: bool = False) -> None:
        self._lang = lang
        self._use_gpu = use_gpu
        self._ocr_client = None
        self._is_mock = False

    @property
    def name(self) -> str:
        return "paddle_ocr"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(self) -> None:
        if PDOCR is None:
            logger.warn(
                "PaddleOCR is not installed. OCR capabilities will run in MOCK mode."
            )
            self._is_mock = True
            return

        try:
            # Initialize PaddleOCR client (downloads models if not present)
            self._ocr_client = PDOCR(
                use_angle_cls=True,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.info("PaddleOCR client initialized successfully")
        except Exception as e:
            logger.error(
                "Failed to initialize PaddleOCR client, running in MOCK mode",
                error=str(e),
            )
            self._is_mock = True

    async def extract_text(self, image_path: Path) -> str:
        regions = await self.extract_regions(image_path)
        return "\n".join([r["text"] for r in regions])

    async def extract_regions(self, image_path: Path) -> list[dict[str, Any]]:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for OCR: {image_path}")

        if self._is_mock or not self._ocr_client:
            logger.info("Using mock OCR fallback for image", path=str(image_path))
            stem = image_path.stem
            return [
                {
                    "text": f"Mock OCR Text detected from image {stem}.",
                    "bbox": [[50.0, 50.0], [450.0, 50.0], [450.0, 80.0], [50.0, 80.0]],
                    "confidence": 0.99,
                },
                {
                    "text": (
                        "This is a paragraph of mock OCR content representing "
                        "layout detection verification."
                    ),
                    "bbox": [
                        [50.0, 100.0],
                        [500.0, 100.0],
                        [500.0, 140.0],
                        [50.0, 140.0],
                    ],
                    "confidence": 0.95,
                },
                {
                    "text": "Table row 1: Item A | Price $10.00 | Qty 5",
                    "bbox": [
                        [50.0, 180.0],
                        [400.0, 180.0],
                        [400.0, 200.0],
                        [50.0, 200.0],
                    ],
                    "confidence": 0.92,
                },
                {
                    "text": "Table row 2: Item B | Price $20.00 | Qty 2",
                    "bbox": [
                        [50.0, 210.0],
                        [400.0, 210.0],
                        [400.0, 230.0],
                        [50.0, 230.0],
                    ],
                    "confidence": 0.93,
                },
            ]

        try:
            # Run OCR on the image file
            result = self._ocr_client.ocr(str(image_path), cls=True)
            if not result or not result[0]:
                return []

            regions: list[dict[str, Any]] = []
            for line in result[0]:
                bbox = line[
                    0
                ]  # List of 4 points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text, confidence = line[1]
                regions.append(
                    {"text": text, "bbox": bbox, "confidence": float(confidence)}
                )
            return regions

        except Exception as e:
            logger.error(
                "Error running PaddleOCR extraction", path=str(image_path), error=str(e)
            )
            return []
