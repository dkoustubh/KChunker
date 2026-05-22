from pathlib import Path
from typing import Any

from plugin_system.base import BaseParser, ExtractedDocument
from utils.logging import logger

try:
    import pandas as pd
except ImportError:
    pd = None


class ExcelParser(BaseParser):
    """Parses Excel spreadsheets (XLSX, XLS, CSV) into structured tables and text."""

    @property
    def name(self) -> str:
        return "excel_parser"

    @property
    def version(self) -> str:
        return "1.0.0"

    def can_parse(self, file_path: Path, mime_type: str) -> bool:
        suffix = file_path.suffix.lower()
        return (
            suffix in [".xlsx", ".xls", ".csv"]
            or "excel" in mime_type
            or "sheet" in mime_type
            or "csv" in mime_type
        )

    async def parse(self, file_path: Path) -> ExtractedDocument:
        logger.info("Parsing Excel/Spreadsheet document", path=str(file_path))

        raw_text_parts: list[str] = []
        extracted_tables: list[dict[str, Any]] = []
        sheet_names: list[str] = []

        if pd is None:
            raise ImportError(
                "Pandas is required for parsing excel files. "
                "Please run `uv sync` to install dependencies."
            )

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".csv":
                # Handle CSV
                df = pd.read_csv(file_path)
                df = df.fillna("")
                data_list = [df.columns.tolist(), *df.values.tolist()]

                # Clean elements
                cleaned_data = [
                    [str(cell).strip() for cell in row] for row in data_list
                ]

                sheet_name = "Sheet1"
                sheet_names.append(sheet_name)
                extracted_tables.append(
                    {"sheet_name": sheet_name, "data": cleaned_data}
                )

                # Build text representation
                markdown_text = df.to_markdown(index=False) or ""
                raw_text_parts.append(f"Sheet: {sheet_name}\n\n{markdown_text}")

            else:
                # Handle Excel (.xlsx, .xls)
                excel_file = pd.ExcelFile(file_path)
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    df = df.fillna("")

                    data_list = [df.columns.tolist(), *df.values.tolist()]
                    cleaned_data = [
                        [str(cell).strip() for cell in row] for row in data_list
                    ]

                    sheet_names.append(sheet_name)
                    extracted_tables.append(
                        {"sheet_name": sheet_name, "data": cleaned_data}
                    )

                    # Convert to markdown text for RAG search
                    try:
                        markdown_text = df.to_markdown(index=False) or ""
                    except Exception:
                        # Fallback if tabulate is not available
                        markdown_text = df.to_string(index=False)

                    raw_text_parts.append(f"Sheet: {sheet_name}\n\n{markdown_text}")

        except Exception as e:
            logger.error(
                "Failed to parse Excel file", path=str(file_path), error=str(e)
            )
            raise e

        combined_text = "\n\n========================================\n\n".join(
            raw_text_parts
        )

        return ExtractedDocument(
            document_name=file_path.name,
            document_path=str(file_path),
            document_type="EXCEL",
            raw_text=combined_text,
            pages=[{"page_num": 1, "text": combined_text}],
            tables=extracted_tables,
            emails=[],
            metadata={"sheets": sheet_names, "total_sheets": len(sheet_names)},
        )
