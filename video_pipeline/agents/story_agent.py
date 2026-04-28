"""
Story Agent: turns a niche/topic into a structured story.
"""
from agents.base import BaseAgent
from tools.llm_client import get_llm
from state import Story


class StoryAgent(BaseAgent):
    name = "StoryAgent"

    SYSTEM = """You are a master cinematic storyteller.
Given a niche/topic, write a compelling 3-act short film story (3-5 minutes screen time).
Focus on emotional resonance, vivid imagery, and a clear narrative arc."""

    USER_TPL = """Niche/topic: {niche}

Generate a story with this exact JSON schema:
{{
  "title": "string - cinematic title",
  "logline": "string - one-sentence summary",
  "synopsis": "string - 3-5 sentence overview",
  "acts": [
    {{"title": "Act 1: Setup", "summary": "what happens in act 1"}},
    {{"title": "Act 2: Confrontation", "summary": "what happens in act 2"}},
    {{"title": "Act 3: Resolution", "summary": "what happens in act 3"}}
  ]
}}"""

    async def run(self, state):
        llm = get_llm()
        data = llm.complete_json(
            system=self.SYSTEM,
            user=self.USER_TPL.format(niche=state.niche),
        )
        state.story = Story(
            niche=state.niche,
            title=data["title"],
            logline=data["logline"],
            synopsis=data["synopsis"],
            acts=data["acts"],
        )
        self.logger.info(f"Story created: {state.story.title}")
        return state
