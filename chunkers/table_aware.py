from typing import Any

from plugin_system.base import Chunk


class TableAwareChunker:
    """Processes structured tables, converting them to Markdown format while avoiding split rows."""

    def __init__(
        self, max_rows_per_chunk: int = 15, max_rows: int | None = None
    ) -> None:
        self.max_rows_per_chunk = (
            max_rows if max_rows is not None else max_rows_per_chunk
        )

    def chunk_table(
        self,
        table_dict: dict[str, Any],
        doc_name: str,
        doc_type: str,
        start_counter: int = 0,
    ) -> tuple[list[Chunk], int]:
        """Converts an extracted table into one or more linked Markdown chunks.

        Table dict format:
        {
            "page": int,
            "table_index": int,
            "sheet_name": str (optional),
            "data": List[List[str]]
        }
        """
        data: list[list[str]] = table_dict.get("data", [])
        if not data:
            return [], start_counter

        page = table_dict.get("page")
        table_idx = table_dict.get("table_index", 0)
        sheet_name = table_dict.get("sheet_name")

        header = data[0]
        rows = data[1:]

        # If the table has no rows other than header, just format the header
        if not rows:
            md_text = self._to_markdown([header])
            chunk = Chunk(
                chunk_id=f"{doc_name}_table_{table_idx}_0",
                document_name=doc_name,
                document_type=doc_type,
                chunk_type="TABLE",
                page=page,
                section=sheet_name or f"Table {table_idx}",
                content=md_text,
                metadata={"sheet_name": sheet_name, "is_table": True},
            )
            return [chunk], start_counter + 1

        chunks: list[Chunk] = []
        chunk_counter = start_counter

        # Group rows into sub-tables of max_rows_per_chunk size
        sub_tables: list[list[list[str]]] = []
        for i in range(0, len(rows), self.max_rows_per_chunk):
            sub_rows = rows[i : i + self.max_rows_per_chunk]
            sub_tables.append([header, *sub_rows])

        sub_chunk_ids = []

        # Phase 1: Create all sub-table chunks
        for idx, sub_table in enumerate(sub_tables):
            chunk_id = f"{doc_name}_table_{table_idx}_p{page or 1}_{chunk_counter}"
            sub_chunk_ids.append(chunk_id)
            chunk_counter += 1

            md_text = self._to_markdown(sub_table)

            # Format sheet/table reference in context
            context_prefix = ""
            if sheet_name:
                context_prefix = f"Sheet: {sheet_name}\n"
            if len(sub_tables) > 1:
                context_prefix += f"Table Part {idx + 1} of {len(sub_tables)}\n"

            full_content = context_prefix + md_text

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_name=doc_name,
                    document_type=doc_type,
                    chunk_type="TABLE",
                    page=page,
                    section=sheet_name or f"Table {table_idx}",
                    content=full_content,
                    metadata={
                        "sheet_name": sheet_name,
                        "table_index": table_idx,
                        "part_index": idx,
                        "total_parts": len(sub_tables),
                        "is_table": True,
                    },
                )
            )

        # Phase 2: Link all parts of the split table together
        if len(chunks) > 1:
            for chunk in chunks:
                chunk.related_chunks = [
                    cid for cid in sub_chunk_ids if cid != chunk.chunk_id
                ]

        return chunks, chunk_counter

    def _to_markdown(self, table_data: list[list[str]]) -> str:
        """Formats a list of lists into a standard Markdown table."""
        if not table_data:
            return ""

        lines = []

        # Header Row
        header_row = " | ".join(str(cell).strip() for cell in table_data[0])
        lines.append(f"| {header_row} |")

        # Separator Row
        separator_row = " | ".join("---" for _ in range(len(table_data[0])))
        lines.append(f"| {separator_row} |")

        # Data Rows
        for row in table_data[1:]:
            # Ensure row has matching columns
            row_cells = list(row) + [""] * (len(table_data[0]) - len(row))
            row_str = " | ".join(
                str(cell).strip() for cell in row_cells[: len(table_data[0])]
            )
            lines.append(f"| {row_str} |")

        return "\n".join(lines)
