from __future__ import annotations

from pathlib import Path


def get_message_classifier_prompt(message: str) -> str:
    """Return the prompt template used for support-message classification."""
    prompt_path = Path(__file__).with_name("message_classifier_prompt.txt")
    template = prompt_path.read_text(encoding="utf-8")
    return f"{template}\n\nMessage: {message}"
