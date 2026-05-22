from parsers.docx_parser import DOCXParser
from parsers.email_parser import EmailParser
from parsers.excel_parser import ExcelParser
from parsers.ocr_parser import OCRParser
from parsers.pdf_parser import PDFParser
from parsers.router import ParserRouter
from parsers.txt_parser import TXTParser

__all__ = [
    "DOCXParser",
    "EmailParser",
    "ExcelParser",
    "OCRParser",
    "PDFParser",
    "ParserRouter",
    "TXTParser",
]
