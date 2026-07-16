from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_supported_file(path: str | Path, supported_extensions: tuple[str, ...]) -> bool:
    """Check whether the supplied filename has a supported extension."""
    return Path(path).suffix.lower() in supported_extensions


def clean_text(text: str) -> str:
    """Normalize whitespace and newlines in extracted text."""
    if not text:
        return ""
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return "\n".join(part for part in normalized.split("\n") if part.strip())


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a Python object as JSON."""
    path = ensure_directory(path.parent if hasattr(path, "parent") else Path(path)) / Path(path).name if isinstance(path, Path) else Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON content from disk."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
