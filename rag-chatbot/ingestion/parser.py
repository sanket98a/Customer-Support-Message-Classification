from __future__ import annotations

import re

from langchain_core.documents import Document

from utils.helper import clean_text


class DocumentParser:
    """Clean and normalize document text before chunking."""

    def __init__(self) -> None:
        self.logger = None

    def parse(self, document: Document) -> Document:
        """Normalize text, remove common header/footer noise, and retain metadata."""
        cleaned_text = self._clean_text(document.page_content)
        return Document(page_content=cleaned_text, metadata={**document.metadata})

    def _clean_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_lines: list[str] = []
        for line in lines:
            if re.match(r"^(page|p\.)\s*\d+$", line.lower()):
                continue
            if line.lower() in {"copyright", "all rights reserved"}:
                continue
            cleaned_lines.append(re.sub(r"\s+", " ", line))
        return clean_text("\n".join(cleaned_lines))
