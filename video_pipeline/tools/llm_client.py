"""
LLM client wrapper - supports Anthropic Claude and OpenAI.
Provides one interface so agents don't care about the provider.

Anthropic prompt caching is enabled when ENABLE_PROMPT_CACHE is set: the
system block is marked `cache_control: ephemeral`, which on Claude 4.x cuts
input cost by ~90% on repeated system prompts (every agent re-uses the same
SYSTEM string across scenes).
"""
import json
import logging
import re
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

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        """Send a completion request and return raw text.

        `model` override lets simple agents (PromptEngineer) use the cheaper
        LLM_FAST_MODEL instead of the default LLM_MODEL.
        """
        chosen_model = model or config.LLM_MODEL
        if self.provider == "anthropic":
            # Cache the system prompt — it's the largest stable chunk and is
            # repeated across every scene/character call from the same agent.
            if config.ENABLE_PROMPT_CACHE:
                system_param = [{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                system_param = system
            response = self._client.messages.create(
                model=chosen_model,
                max_tokens=max_tokens,
                system=system_param,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        elif self.provider == "openai":
            response = self._client.chat.completions.create(
                model=chosen_model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content
        return ""

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> dict | list:
        """Send a request and parse the response as JSON."""
        system_json = system + (
            "\n\nIMPORTANT: Respond with ONLY valid JSON. "
            "No prose, no markdown fences, no preamble."
        )
        raw = self.complete(system_json, user, max_tokens, model=model)
        return _parse_json_lenient(raw)


def _parse_json_lenient(raw: str) -> dict | list:
    """Best-effort JSON extraction. Strips fences, finds first {...} or [...]."""
    cleaned = raw.strip()
    # Strip ```json ... ``` style fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # drop first and (possibly) last fence lines
        if lines[-1].startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back: grab the first balanced JSON object/array via regex.
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        logger.error(f"JSON parse failed. Raw output: {raw[:500]}")
        raise


# Lazy singleton
_llm_instance: Optional[LLMClient] = None

def get_llm() -> LLMClient:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance
