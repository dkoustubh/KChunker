import re
from email import message_from_file
from email.policy import default
from pathlib import Path
from typing import Any

from plugin_system.base import BaseParser, ExtractedDocument
from utils.logging import logger


class EmailParser(BaseParser):
    """Parses emails (.eml, .msg) and splits chains into individual messages."""

    @property
    def name(self) -> str:
        return "email_parser"

    @property
    def version(self) -> str:
        return "1.0.0"

    def can_parse(self, file_path: Path, mime_type: str) -> bool:
        suffix = file_path.suffix.lower()
        return (
            suffix in [".eml", ".msg"]
            or "message" in mime_type
            or "rfc822" in mime_type
        )

    async def parse(self, file_path: Path) -> ExtractedDocument:
        logger.info("Parsing Email document", path=str(file_path))

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                msg = message_from_file(f, policy=default)

            subject = msg.get("Subject", "No Subject")
            from_addr = msg.get("From", "No Sender")
            to_addr = msg.get("To", "No Recipient")
            date_str = msg.get("Date", "")

            # Extract body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if (
                        content_type == "text/plain"
                        and "attachment" not in content_disposition
                    ):
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body = payload.decode(errors="ignore")
                        else:
                            body = str(payload or "")
                        break
                    elif (
                        content_type == "text/html"
                        and "attachment" not in content_disposition
                    ):
                        payload_html = part.get_payload(decode=True)
                        if isinstance(payload_html, bytes):
                            body_html = payload_html.decode(errors="ignore")
                        else:
                            body_html = str(payload_html or "")
                        # Simple HTML tags removal fallback
                        body = re.sub(r"<[^>]+>", "", body_html)
            else:
                payload = msg.get_payload(decode=True)
                if isinstance(payload, bytes):
                    body = payload.decode(errors="ignore")
                else:
                    body = str(payload or "")

            # Parse email trail/chain
            individual_emails = self._split_email_trail(
                body, subject, from_addr, date_str
            )

            # Combined metadata text representation
            meta_header = (
                f"Subject: {subject}\nFrom: {from_addr}\n"
                f"To: {to_addr}\nDate: {date_str}\n\n"
            )
            combined_text = meta_header + body

        except Exception as e:
            logger.error(
                "Failed to parse email file", path=str(file_path), error=str(e)
            )
            raise e

        return ExtractedDocument(
            document_name=file_path.name,
            document_path=str(file_path),
            document_type="EMAIL",
            raw_text=combined_text,
            pages=[{"page_num": 1, "text": combined_text}],
            tables=[],
            emails=individual_emails,
            metadata={
                "subject": subject,
                "from": from_addr,
                "to": to_addr,
                "date": date_str,
                "trail_depth": len(individual_emails),
            },
        )

    def _split_email_trail(
        self, body: str, subject: str, from_addr: str, date: str
    ) -> list[dict[str, Any]]:
        """Splits an email body into a list of messages using separators."""
        # Find header separators or blocks containing "-----Original Message-----"
        # to split text based on headers.
        headers_pattern = re.compile(
            r"(?:\-\-\-\-\-\s*Original Message\s*\-\-\-\-\-|"
            r"\-\-\-\-\-\s*Forwarded message\s*\-\-\-\-\-|"
            r"From:\s*[^\n]+\n(?:Sent|Date):\s*[^\n]+\nTo:\s*[^\n]+)"
        )

        splits = headers_pattern.split(body)
        headers_found = headers_pattern.findall(body)

        emails: list[dict[str, Any]] = []

        # First section is the most recent email
        recent_body = splits[0].strip()
        emails.append(
            {
                "index": 0,
                "sender": from_addr,
                "date": date,
                "subject": subject,
                "body": recent_body,
            }
        )

        for i, block in enumerate(splits[1:]):
            header_context = headers_found[i] if i < len(headers_found) else ""

            # Extract header values if possible
            sender_match = re.search(r"From:\s*([^\n]+)", header_context, re.IGNORECASE)
            date_match = re.search(
                r"(?:Sent|Date):\s*([^\n]+)", header_context, re.IGNORECASE
            )
            subj_match = re.search(
                r"Subject:\s*([^\n]+)", header_context, re.IGNORECASE
            )

            emails.append(
                {
                    "index": i + 1,
                    "sender": (
                        sender_match.group(1).strip()
                        if sender_match
                        else "Unknown Sender"
                    ),
                    "date": (
                        date_match.group(1).strip() if date_match else "Unknown Date"
                    ),
                    "subject": (
                        subj_match.group(1).strip() if subj_match else f"Re: {subject}"
                    ),
                    "body": block.strip(),
                }
            )

        return emails
