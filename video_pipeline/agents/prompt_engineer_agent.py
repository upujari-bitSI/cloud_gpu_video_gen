"""
Prompt Engineering Agent: turns scene specs into cinematic image-gen prompts.
"""
from agents.base import BaseAgent
from tools.llm_client import get_llm


CINEMATIC_SUFFIX = (
    "cinematic, photorealistic, ultra-detailed, 8k, depth of field, "
    "shot on ARRI Alexa, 35mm anamorphic lens, film grain, "
    "professional color grading, volumetric lighting, sharp focus"
)


class PromptEngineerAgent(BaseAgent):
    name = "PromptEngineerAgent"

    SYSTEM = """You are an expert prompt engineer for SDXL/cinematic AI image generation.
Convert a scene description into a single dense, comma-separated visual prompt.
Use concrete visual nouns and adjectives. Emphasize: subject, environment, lighting, lens, mood.
Do NOT use abstract concepts. Maximum 80 words."""

    USER_TPL = """Scene:
Title: {title}
Environment: {environment}
Mood: {mood}
Camera: {camera}
Lighting: {lighting}
Action: {actions}

Characters present:
{character_visuals}

Write ONE prompt string (no JSON, no markdown, just the prompt text)."""

    async def run(self, state):
        if not state.scenes:
            raise ValueError("No scenes to engineer prompts for.")

        char_lookup = {c.name: c.visual_prompt for c in state.characters}
        llm = get_llm()

        for scene in state.scenes:
            char_visuals = "\n".join(
                f"- {name}: {char_lookup.get(name, 'no visual spec')}"
                for name in scene.characters
            ) or "- (none)"

            prompt_text = llm.complete(
                system=self.SYSTEM,
                user=self.USER_TPL.format(
                    title=scene.title,
                    environment=scene.environment,
                    mood=scene.mood,
                    camera=scene.camera,
                    lighting=scene.lighting,
                    actions=scene.actions,
                    character_visuals=char_visuals,
                ),
                max_tokens=300,
            ).strip()

            scene.image_prompt = f"{prompt_text}, {CINEMATIC_SUFFIX}"
            self.logger.info(f"Scene {scene.index}: {scene.image_prompt[:90]}...")

        return state
