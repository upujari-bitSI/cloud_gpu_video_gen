"""
Prompt Engineering Agent: turns scene specs into image-gen prompts.

Two efficiency wins vs the original:
1. All scenes are sent to the LLM in a SINGLE batched JSON request
   (was: one HTTP round-trip per scene). 6-30x fewer Claude calls.
2. Uses LLM_FAST_MODEL (Haiku) — prompt-engineering is a simple template
   task and Sonnet/Opus is overkill.
"""
import json
from agents.base import BaseAgent
from tools.llm_client import get_llm
from config import config


class PromptEngineerAgent(BaseAgent):
    name = "PromptEngineerAgent"

    SYSTEM = """You are an expert prompt engineer for SDXL/cocomelon-style AI image generation.
For each scene, write ONE dense comma-separated visual prompt.
Use concrete visual nouns and adjectives. Emphasize subject, environment,
lighting, lens, mood. Maximum 60 words per prompt.
Always include the character's full visual description verbatim when that
character is in the scene — this keeps the character looking identical
across every shot.
Return JSON: a list of strings, one per scene, in the same order as input."""

    USER_TPL = """Style suffix that will be appended automatically (do NOT include it):
{style_suffix}

Character visual specs (use verbatim when the character appears):
{character_specs}

Scenes (in order):
{scenes_json}

Return JSON ONLY. Format:
["prompt for scene 0", "prompt for scene 1", ...]"""

    async def run(self, state):
        if not state.scenes:
            raise ValueError("No scenes to engineer prompts for.")

        char_lookup = {c.name: c.visual_prompt for c in state.characters}
        style = config.style()
        style_suffix = style["suffix"]

        scene_payload = [
            {
                "index": s.index,
                "title": s.title,
                "environment": s.environment,
                "mood": s.mood,
                "camera": s.camera,
                "lighting": s.lighting,
                "actions": s.actions,
                "characters": s.characters,
            }
            for s in state.scenes
        ]
        character_specs = "\n".join(
            f"- {name}: {visual}" for name, visual in char_lookup.items()
        ) or "- (no named characters)"

        llm = get_llm()
        prompts = llm.complete_json(
            system=self.SYSTEM,
            user=self.USER_TPL.format(
                style_suffix=style_suffix,
                character_specs=character_specs,
                scenes_json=json.dumps(scene_payload, indent=2),
            ),
            max_tokens=4096,
            model=config.LLM_FAST_MODEL,
        )
        if isinstance(prompts, dict) and "prompts" in prompts:
            prompts = prompts["prompts"]
        if not isinstance(prompts, list) or len(prompts) != len(state.scenes):
            self.logger.warning(
                f"Prompt batch returned {len(prompts) if isinstance(prompts, list) else '?'} "
                f"items for {len(state.scenes)} scenes; falling back to per-scene calls."
            )
            await self._per_scene_fallback(state, char_lookup, style_suffix)
            return state

        for scene, p in zip(state.scenes, prompts):
            scene.image_prompt = f"{p.strip()}, {style_suffix}"
            self.logger.info(f"Scene {scene.index}: {scene.image_prompt[:90]}...")
        return state

    async def _per_scene_fallback(self, state, char_lookup, style_suffix):
        """Used only if batched call fails — keeps the pipeline moving."""
        llm = get_llm()
        for scene in state.scenes:
            char_visuals = "\n".join(
                f"- {name}: {char_lookup.get(name, 'no visual spec')}"
                for name in scene.characters
            ) or "- (none)"
            user = (
                f"Title: {scene.title}\nEnvironment: {scene.environment}\n"
                f"Mood: {scene.mood}\nCamera: {scene.camera}\n"
                f"Lighting: {scene.lighting}\nAction: {scene.actions}\n"
                f"Characters:\n{char_visuals}\n\nWrite ONE prompt string."
            )
            prompt_text = llm.complete(
                system="You write concise SDXL image prompts. Return ONLY the prompt string.",
                user=user,
                max_tokens=300,
                model=config.LLM_FAST_MODEL,
            ).strip()
            scene.image_prompt = f"{prompt_text}, {style_suffix}"
