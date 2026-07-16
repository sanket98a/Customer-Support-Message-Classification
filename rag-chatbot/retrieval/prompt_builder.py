from __future__ import annotations

from pathlib import Path
from typing import Sequence

from config.config import AppConfig


def build_prompt(question: str, contexts: Sequence[dict[str, object]], system_prompt: str | None = None) -> str:
    """Create a prompt that instructs the model to use only retrieved context."""
    system_text = system_prompt or Path(AppConfig().system_prompt_path).read_text(encoding="utf-8")

    context_sections: list[str] = []
    for index, context in enumerate(contexts, start=1):
        chunk = context.get("chunk", {})
        document_name = chunk.get("document_name", "Unknown")
        page_number = chunk.get("page_number")
        page_suffix = f"Page {page_number}" if page_number is not None else "Page unknown"
        context_sections.append(
            f"Source {index}: {document_name} ({page_suffix})\n{chunk.get('text', '')}"
        )

    retrieved_context = "\n\n".join(context_sections) if context_sections else "No context was retrieved."
    return (
        f"{system_text}\n\n"
        f"Retrieved Context:\n{retrieved_context}\n\n"
        f"User Question:\n{question}\n\n"
        "Answer:"
    )
