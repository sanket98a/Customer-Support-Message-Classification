from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config.config import AppConfig
from ingestion.chunker import Chunk
from ingestion.embedding import OllamaEmbeddingService
from utils.helper import ensure_directory, load_json, save_json
from utils.logger import get_logger


class FaissIndexer:
    """Create and manage a local FAISS index for retrieved chunks."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.logger = get_logger("FaissIndexer")
        ensure_directory(self.config.vector_store_dir)
        ensure_directory(self.config.processed_dir)

    def build_index(self, chunks: list[Chunk], embedding_service: OllamaEmbeddingService) -> None:
        """Generate embeddings for chunks and persist the index."""
        if not chunks:
            raise ValueError("No chunks available to index")

        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_service.embed_documents(texts)
        if not embeddings:
            raise ValueError("Ollama did not return embeddings")

        vector_matrix = np.array(embeddings, dtype=np.float32)
        dimension = vector_matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vector_matrix)
        faiss.write_index(index, str(self.config.faiss_path))

        payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "chunks": [chunk.to_dict() for chunk in chunks],
        }
        save_json(self.config.metadata_path, payload)
        save_json(self.config.chunk_store_path, payload)
        self.logger.info("Saved FAISS index to %s", self.config.faiss_path)

    def is_index_available(self) -> bool:
        """Return True when the FAISS index exists."""
        return self.config.faiss_path.exists() and self.config.metadata_path.exists()

    def load_index(self) -> tuple[faiss.Index, list[dict[str, object]]]:
        """Load the FAISS index and metadata from disk."""
        if not self.is_index_available():
            raise FileNotFoundError("FAISS index is missing. Build the index first.")

        index = faiss.read_index(str(self.config.faiss_path))
        payload = load_json(self.config.metadata_path)
        chunks = payload.get("chunks", [])
        return index, chunks

    def search(self, query: str, embedding_service: OllamaEmbeddingService, top_k: int | None = None) -> list[dict[str, object]]:
        """Search the FAISS index for the most relevant chunks."""
        index, chunks = self.load_index()
        query_embedding = embedding_service.embed_query(query)
        if not query_embedding:
            raise ValueError("Unable to create query embedding")

        query_vector = np.array([query_embedding], dtype=np.float32)
        limit = top_k or self.config.top_k
        distances, indices = index.search(query_vector, limit)

        results: list[dict[str, object]] = []
        for distance, index_position in zip(distances[0], indices[0]):
            if index_position < 0 or index_position >= len(chunks):
                continue
            chunk = chunks[int(index_position)]
            results.append({
                "score": float(distance),
                "chunk": chunk,
            })
        return results

    def clear(self) -> None:
        """Delete index artifacts from disk."""
        for path in [self.config.faiss_path, self.config.metadata_path, self.config.chunk_store_path]:
            if path.exists():
                path.unlink()
        self.logger.info("Removed FAISS artifacts")
