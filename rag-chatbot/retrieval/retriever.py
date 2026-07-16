from __future__ import annotations

from typing import Optional

from config.config import AppConfig
from ingestion.embedding import OllamaEmbeddingService
from ingestion.indexer import FaissIndexer
from utils.logger import get_logger


class DocumentRetriever:
    """Retrieve relevant chunks from the local FAISS index."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.indexer = FaissIndexer(config)
        self.embedding_service = OllamaEmbeddingService(config)
        self.logger = get_logger("DocumentRetriever")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, object]]:
        """Return top-k matching chunks with their similarity scores."""
        if not self.indexer.is_index_available():
            raise FileNotFoundError("FAISS index is missing. Please build the index first.")
        return self.indexer.search(query, self.embedding_service, top_k=top_k)
