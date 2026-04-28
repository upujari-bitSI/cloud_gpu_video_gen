"""
Base Agent class. All agents inherit from this.
"""
import logging
import asyncio
from abc import ABC, abstractmethod
from config import config


class BaseAgent(ABC):
    """Abstract base for all agents in the pipeline."""

    name: str = "BaseAgent"

    def __init__(self):
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    async def run(self, state):
        """Execute the agent on the shared pipeline state and return updated state."""
        ...

    async def run_with_retry(self, state, max_retries: int = None):
        """Run with automatic retry on failure."""
        max_retries = max_retries or config.MAX_RETRIES
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"[{self.name}] attempt {attempt}/{max_retries}")
                return await self.run(state)
            except Exception as e:
                last_err = e
                self.logger.warning(f"[{self.name}] failed attempt {attempt}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
        state.errors.append(f"{self.name}: {last_err}")
        raise last_err
