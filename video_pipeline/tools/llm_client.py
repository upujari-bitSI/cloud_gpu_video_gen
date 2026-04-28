"""
LLM client wrapper - supports Anthropic Claude and OpenAI.
Provides one interface so agents don't care about the provider.
"""
import json
import logging
from typing import Optional
from config import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client with JSON response parsing."""

    def __init__(self):
        self.provider = config.LLM_PROVIDER
        self._client = None

        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
            except ImportError:
                raise ImportError("Run: pip install anthropic")
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=config.OPENAI_API_KEY)
            except ImportError:
                raise ImportError("Run: pip install openai")
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Send a completion request and return raw text."""
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        elif self.provider == "openai":
            response = self._client.chat.completions.create(
                model=config.LLM_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content
        return ""

    def complete_json(self, system: str, user: str, max_tokens: int = 4096) -> dict | list:
        """Send a request and parse the response as JSON."""
        # Append explicit JSON-only instruction
        system_json = system + "\n\nIMPORTANT: Respond with ONLY valid JSON. No prose, no markdown fences, no preamble."
        raw = self.complete(system_json, user, max_tokens)
        # Strip common markdown fences if model adds them anyway
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}\nRaw output: {raw[:500]}")
            raise


# Lazy singleton
_llm_instance: Optional[LLMClient] = None

def get_llm() -> LLMClient:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance
