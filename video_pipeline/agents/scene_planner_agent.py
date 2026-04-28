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
Convert a story into a sequence of detailed scenes for a cinematic short video.
Pace scenes so total run-time matches the requested target duration.
Each scene should have its own visual idea — do not repeat shots."""

    USER_TPL = """Story Title: {title}
Logline: {logline}
Synopsis: {synopsis}

Acts:
{acts}

TARGET TOTAL RUN-TIME: {target_seconds:.0f} seconds.
Plan approximately {target_count} scenes, each {avg_duration:.1f} seconds long
(narration of 2-3 short sentences fits this duration). The narration text for
each scene must be readable aloud in roughly that duration; do NOT write more
text than will fit.

Return JSON array (or {{"scenes": [...]}}):
[
  {{
    "title": "Scene name",
    "narration": "voice-over text (2-3 short sentences)",
    "environment": "where the scene takes place, vivid details",
    "mood": "emotional tone (hopeful, tense, melancholic, etc.)",
    "camera": "shot type and movement (wide establishing, slow dolly in, low angle, etc.)",
    "lighting": "lighting style (golden hour, harsh fluorescent, neon, candlelit, etc.)",
    "characters": ["list of character names present"],
    "actions": "what characters are doing physically",
    "duration": {avg_duration:.1f}
  }}
]"""

    async def run(self, state):
        if not state.story:
            raise ValueError("Story must be generated before scene planning.")

        # Aim for ~10s per scene as a reasonable narration unit. Both the
        # actual count and per-scene duration get re-tuned later by the
        # VoiceOver agent based on real TTS audio length.
        avg_duration = 10.0
        target_count = max(6, int(round(config.TARGET_DURATION_SECONDS / avg_duration)))

        acts_text = "\n".join(f"- {a['title']}: {a['summary']}" for a in state.story.acts)
        llm = get_llm()
        scenes_data = llm.complete_json(
            system=self.SYSTEM,
            user=self.USER_TPL.format(
                title=state.story.title,
                logline=state.story.logline,
                synopsis=state.story.synopsis,
                acts=acts_text,
                target_seconds=config.TARGET_DURATION_SECONDS,
                target_count=target_count,
                avg_duration=avg_duration,
            ),
            max_tokens=8192,
        )
        if isinstance(scenes_data, dict) and "scenes" in scenes_data:
            scenes_data = scenes_data["scenes"]

        state.scenes = []
        for i, sd in enumerate(scenes_data):
            duration = float(sd.get("duration", avg_duration))
            duration = max(config.MIN_SCENE_DURATION,
                           min(config.MAX_SCENE_DURATION, duration))
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
                duration=duration,
            ))
        total = sum(s.duration for s in state.scenes)
        self.logger.info(
            f"Planned {len(state.scenes)} scenes, ~{total:.0f}s total "
            f"(target {config.TARGET_DURATION_SECONDS:.0f}s)."
        )
        return state
