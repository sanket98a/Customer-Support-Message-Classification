from __future__ import annotations

from typing import Optional

import requests

from config.config import AppConfig
from utils.logger import get_logger


class OllamaGenerator:
    """Generate answers through the local Ollama API."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.logger = get_logger("OllamaGenerator")
        self.base_url = self.config.ollama_base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the generated response."""
        payload = {
            "model": self.config.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
