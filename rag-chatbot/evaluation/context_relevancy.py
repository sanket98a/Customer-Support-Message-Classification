from __future__ import annotations

import re
from typing import Sequence


def evaluate_context_relevancy(question: str, chunks: Sequence[dict[str, object]]) -> float:
    """Score context relevancy with a simple lexical overlap heuristic."""
    if not chunks:
        return 0.0

    question_terms = {term.lower() for term in re.findall(r"\w+", question) if term}
    if not question_terms:
        return 0.0

    relevant_count = 0
    for chunk in chunks:
        chunk_text = str(chunk.get("text", "")).lower()
        chunk_terms = {term for term in re.findall(r"\w+", chunk_text) if term}
        if question_terms & chunk_terms:
            relevant_count += 1

    return round(relevant_count / len(chunks), 2)
