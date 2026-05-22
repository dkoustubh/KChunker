from typing import Any

from pydantic import BaseModel, Field


class LayoutBlock(BaseModel):
    """Represents a coordinate-aware layout block within a document page."""

    block_id: str
    block_type: str = Field(description="TEXT, HEADER, TABLE, LIST")
    content: str
    page_num: int
    coordinates: list[float] = Field(
        default_factory=list, description="[x0, y0, x1, y1] bounding box"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class LayoutDetector:
    """Analyzes document layout structure to group segments into blocks."""

    @staticmethod
    def detect_layout_from_blocks(
        pymupdf_blocks: list[tuple[Any, ...]], page_num: int
    ) -> list[LayoutBlock]:
        """Converts PyMuPDF block tuple outputs into normalized LayoutBlock structures.

        PyMuPDF block tuple format: (x0, y0, x1, y1, "text", block_no, block_type)
        """
        layout_blocks: list[LayoutBlock] = []
        for idx, block in enumerate(pymupdf_blocks):
            if len(block) >= 5:
                x0, y0, x1, y1, text, block_no, btype = block[:7]
                text = text.strip()
                if not text:
                    continue

                # Classify headers by brief content, list prefixes, or uppercase format.
                block_type = "TEXT"
                lines = text.split("\n")
                if len(lines) == 1 and len(text) < 100 and text.isupper():
                    block_type = "HEADER"
                elif text.startswith(("- ", "* ", "1. ", "• ")):
                    block_type = "LIST"

                layout_blocks.append(
                    LayoutBlock(
                        block_id=f"p{page_num}_b{block_no}_{idx}",
                        block_type=block_type,
                        content=text,
                        page_num=page_num,
                        coordinates=[float(x0), float(y0), float(x1), float(y1)],
                        metadata={"block_no": block_no, "block_type_code": btype},
                    )
                )
        return layout_blocks

    @staticmethod
    def group_ocr_regions(
        regions: list[dict[str, Any]], page_num: int, line_threshold: float = 15.0
    ) -> list[LayoutBlock]:
        """Clusters spatial OCR regions into blocks based on proximity."""
        if not regions:
            return []

        # Sort regions by y-coordinate, then by x-coordinate.
        # Normalize the coordinates from 4-point bounds to flat [x0, y0, x1, y1] bbox.
        normalized_regions = []
        for r in regions:
            bbox = r["bbox"]
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)
            normalized_regions.append(
                {
                    "text": r["text"],
                    "confidence": r.get("confidence", 1.0),
                    "coords": [x0, y0, x1, y1],
                }
            )

        normalized_regions.sort(key=lambda r: (r["coords"][1], r["coords"][0]))

        grouped_blocks: list[LayoutBlock] = []
        if not normalized_regions:
            return grouped_blocks

        # Clustering algorithm
        current_block_regions = [normalized_regions[0]]

        for region in normalized_regions[1:]:
            last_region = current_block_regions[-1]
            last_coords = last_region["coords"]
            curr_coords = region["coords"]

            # Check vertical gap: distance between last bottom and current top
            v_gap = curr_coords[1] - last_coords[3]

            # Check if region is on a new line or close enough vertically to group
            if v_gap < line_threshold:
                current_block_regions.append(region)
            else:
                # Flush the current group
                grouped_blocks.append(
                    LayoutDetector._create_block_from_regions(
                        current_block_regions, page_num, len(grouped_blocks)
                    )
                )
                current_block_regions = [region]

        if current_block_regions:
            grouped_blocks.append(
                LayoutDetector._create_block_from_regions(
                    current_block_regions, page_num, len(grouped_blocks)
                )
            )

        return grouped_blocks

    @staticmethod
    def _create_block_from_regions(
        regions: list[dict[str, Any]], page_num: int, block_idx: int
    ) -> LayoutBlock:
        content = " ".join([r["text"] for r in regions])
        xs = [r["coords"][0] for r in regions] + [r["coords"][2] for r in regions]
        ys = [r["coords"][1] for r in regions] + [r["coords"][3] for r in regions]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)

        block_type = "TEXT"
        if len(regions) == 1 and len(content) < 60 and content.isupper():
            block_type = "HEADER"

        avg_confidence = sum([r["confidence"] for r in regions]) / len(regions)

        return LayoutBlock(
            block_id=f"p{page_num}_ocr_{block_idx}",
            block_type=block_type,
            content=content,
            page_num=page_num,
            coordinates=[x0, y0, x1, y1],
            metadata={"avg_confidence": avg_confidence, "num_segments": len(regions)},
        )
