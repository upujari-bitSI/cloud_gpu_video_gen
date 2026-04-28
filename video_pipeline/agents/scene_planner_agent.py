"""
Scene Planner Agent: breaks the story into shot-level scenes.
"""
from agents.base import BaseAgent
from tools.llm_client import get_llm
from state import Scene
from config import config


class ScenePlannerAgent(BaseAgent):
    name = "ScenePlannerAgent"

    SYSTEM = """You are a film director and storyboard artist.
Convert a story into a sequence of detailed scenes for a short cinematic video.
Each scene should be 4-7 seconds long. Aim for 6-10 scenes total."""

    USER_TPL = """Story Title: {title}
Logline: {logline}
Synopsis: {synopsis}

Acts:
{acts}

Break this story into scenes. Return JSON array:
[
  {{
    "title": "Scene name",
    "narration": "voice-over text the narrator will read (1-2 sentences)",
    "environment": "where the scene takes place, vivid details",
    "mood": "emotional tone (hopeful, tense, melancholic, etc.)",
    "camera": "shot type and movement (wide establishing, slow dolly in, low angle, etc.)",
    "lighting": "lighting style (golden hour, harsh fluorescent, neon, candlelit, etc.)",
    "characters": ["list of character names present"],
    "actions": "what characters are doing physically",
    "duration": 5.0
  }}
]"""

    async def run(self, state):
        if not state.story:
            raise ValueError("Story must be generated before scene planning.")

        acts_text = "\n".join(f"- {a['title']}: {a['summary']}" for a in state.story.acts)
        llm = get_llm()
        scenes_data = llm.complete_json(
            system=self.SYSTEM,
            user=self.USER_TPL.format(
                title=state.story.title,
                logline=state.story.logline,
                synopsis=state.story.synopsis,
                acts=acts_text,
            ),
        )
        if isinstance(scenes_data, dict) and "scenes" in scenes_data:
            scenes_data = scenes_data["scenes"]

        state.scenes = []
        for i, sd in enumerate(scenes_data):
            state.scenes.append(Scene(
                index=i,
                title=sd.get("title", f"Scene {i+1}"),
                narration=sd.get("narration", ""),
                environment=sd.get("environment", ""),
                mood=sd.get("mood", "neutral"),
                camera=sd.get("camera", "medium shot"),
                lighting=sd.get("lighting", "natural"),
                characters=sd.get("characters", []),
                actions=sd.get("actions", ""),
                duration=float(sd.get("duration", config.SCENE_DEFAULT_DURATION)),
            ))
        self.logger.info(f"Planned {len(state.scenes)} scenes.")
        return state
