from __future__ import annotations

from typing import Optional

import requests

from config.config import AppConfig
from utils.logger import get_logger


class OllamaEmbeddingService:
    """Wrap Ollama embeddings for local embedding generation."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.logger = get_logger("OllamaEmbeddingService")
        self.base_url = self.config.ollama_base_url.rstrip("/")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents through Ollama."""
        if not texts:
            return []
        payload = {"model": self.config.embedding_model, "input": texts}
        response = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=120)
        response.raise_for_status()
        payload = response.json()
        return payload.get("embeddings", [])

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        embeddings = self.embed_documents([query])
        if not embeddings:
            return []
        return embeddings[0]
