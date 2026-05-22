from typing import Any

from plugin_system.base import Chunk


class EmailChunker:
    """Chunks email trails, splitting them into individual messages and maintaining thread references."""

    def chunk_emails(
        self,
        emails: list[dict[str, Any]],
        doc_name: str,
        doc_type: str,
        start_counter: int = 0,
    ) -> tuple[list[Chunk], int]:
        """Creates linked chunks for each message in an email trail.

        Email dict format:
        {
            "index": int,
            "sender": str,
            "date": str,
            "subject": str,
            "body": str
        }
        """
        if not emails:
            return [], start_counter

        chunks: list[Chunk] = []
        chunk_counter = start_counter
        email_chunk_ids = []

        # Phase 1: Create chunks for all messages
        for email in emails:
            chunk_id = f"{doc_name}_email_{email.get('index', 0)}_{chunk_counter}"
            email_chunk_ids.append(chunk_id)
            chunk_counter += 1

            sender = email.get("sender", "Unknown Sender")
            date = email.get("date", "Unknown Date")
            subject = email.get("subject", "No Subject")
            body = email.get("body", "")

            formatted_content = (
                f"From: {sender}\n"
                f"Date: {date}\n"
                f"Subject: {subject}\n\n"
                f"{body}"
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_name=doc_name,
                    document_type=doc_type,
                    chunk_type="EMAIL",
                    page=1,
                    section=f"Email {email.get('index', 0)}",
                    content=formatted_content,
                    metadata={
                        "sender": sender,
                        "date": date,
                        "subject": subject,
                        "email_index": email.get("index", 0),
                        "is_email": True,
                    },
                )
            )

        # Phase 2: Link all emails in the same trail together
        if len(chunks) > 1:
            for chunk in chunks:
                chunk.related_chunks = [
                    cid for cid in email_chunk_ids if cid != chunk.chunk_id
                ]

        return chunks, chunk_counter
