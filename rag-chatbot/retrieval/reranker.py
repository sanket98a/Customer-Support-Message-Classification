from __future__ import annotations

import re
from typing import Iterable


class SimpleReranker:
    """Simple lexical reranker for retrieved chunks."""

    def rerank(self, results: Iterable[dict[str, object]], query: str) -> list[dict[str, object]]:
        """Re-rank retrieved chunks by lexical overlap with the query."""
        query_terms = {term for term in re.findall(r"\w+", query.lower()) if term}
        scored_results: list[tuple[float, dict[str, object]]] = []
        for result in results:
            chunk = result.get("chunk", {})
            text = str(chunk.get("text", "")).lower()
            terms = {term for term in re.findall(r"\w+", text) if term}
            overlap = len(query_terms & terms)
            scored_results.append((overlap, result))
        score_sorted = sorted(scored_results, key=lambda item: item[0], reverse=True)
        return [result for _, result in score_sorted]
