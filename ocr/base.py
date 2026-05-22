from abc import abstractmethod
from pathlib import Path
from typing import Any

from plugin_system.base import BasePlugin


class BaseOCR(BasePlugin):
    """Abstract base class for OCR services."""

    @abstractmethod
    async def extract_text(self, image_path: Path) -> str:
        """Extracts plain text from the given image."""
        pass

    @abstractmethod
    async def extract_regions(self, image_path: Path) -> list[dict[str, Any]]:
        """Extracts text along with spatial coordinate regions.

        Returns:
            List of dicts containing 'text' and 'bbox' (coordinates).
        """
        pass
