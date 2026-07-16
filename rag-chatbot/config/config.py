from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    """Central configuration for the local RAG chatbot."""

    project_root: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    vector_store_dir: Path = BASE_DIR / "vector_store" / "faiss_index"
    faiss_path: Path = vector_store_dir / "faiss.index"
    metadata_path: Path = vector_store_dir / "metadata.json"
    chunk_store_path: Path = processed_dir / "chunks.json"
    system_prompt_path: Path = BASE_DIR / "prompts" / "system_prompt.txt"

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text-v2-moe:latest")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "llama3.2:1b")
    temperature: float = float(os.getenv("RAG_TEMPERATURE", "0.1"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    supported_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md", ".markdown")


def get_config() -> AppConfig:
    """Return the application configuration instance."""
    return AppConfig()
