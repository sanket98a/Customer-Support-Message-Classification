from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - fallback for different langchain distributions
    try:
        from langchain_text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
        except Exception:
            raise
from langchain_core.documents import Document

from config.config import AppConfig
from utils.logger import get_logger


@dataclass(slots=True)
class Chunk:
    """A chunk of text with rich metadata for retrieval."""

    chunk_id: str
    text: str
    document_name: str
    page_number: int | None
    section: str
    source_path: str
    chunk_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "section": self.section,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
        }


class Chunker:
    """Split parsed documents into overlapping text chunks."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.logger = get_logger("Chunker")

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into overlapping chunks with metadata."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks: list[Chunk] = []
        for document_index, document in enumerate(documents):
            split_docs = splitter.split_documents([document])
            for chunk_index, split_doc in enumerate(split_docs):
                metadata = split_doc.metadata or {}
                chunk = Chunk(
                    chunk_id=f"{document_index + 1}-{chunk_index + 1}",
                    text=split_doc.page_content,
                    document_name=str(metadata.get("document_name", metadata.get("source", "unknown"))),
                    page_number=self._resolve_page_number(metadata),
                    section=str(metadata.get("section", "General")),
                    source_path=str(metadata.get("file_path", metadata.get("source", "unknown"))),
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
        return chunks

    def _resolve_page_number(self, metadata: dict[str, object]) -> int | None:
        page_number = metadata.get("page_number")
        if page_number is None:
            return None
        return int(page_number)
