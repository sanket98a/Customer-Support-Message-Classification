from __future__ import annotations

import re
from typing import Sequence


def evaluate_groundedness(answer: str, contexts: Sequence[dict[str, object]]) -> float:
    """Return a simple groundedness score based on overlap with retrieved context."""
    if not contexts:
        return 0.0
    context_text = " ".join(str(context.get("chunk", {}).get("text", "")) for context in contexts)
    answer_terms = set(re.findall(r"\w+", answer.lower()))
    context_terms = set(re.findall(r"\w+", context_text.lower()))
    if not answer_terms:
        return 0.0
    overlap = len(answer_terms & context_terms) / len(answer_terms)
    return round(min(overlap, 1.0), 2)
