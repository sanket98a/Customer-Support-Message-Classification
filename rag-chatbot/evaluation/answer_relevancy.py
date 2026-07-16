from __future__ import annotations

import re


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """Return a simple answer relevancy score based on lexical overlap."""
    if not answer:
        return 0.0

    question_terms = {term.lower() for term in re.findall(r"\w+", question) if term}
    answer_terms = {term.lower() for term in re.findall(r"\w+", answer) if term}
    if not question_terms:
        return 0.0
    overlap = len(question_terms & answer_terms) / len(question_terms)
    return round(min(overlap, 1.0), 2)
