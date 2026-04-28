"""
Visual Generation Agent: generates one image per scene using SDXL.

Character consistency: each scene's seed is derived from the primary
character's deterministic seed. Two scenes featuring the same hero will
share a seed, which (combined with the verbatim character visual prompt
that PromptEngineer embeds) keeps the character looking the same shot
to shot. Scenes with no named character fall back to a per-scene seed.
"""
import asyncio
from agents.base import BaseAgent
from tools.image_gen import get_image_generator


class VisualGenerationAgent(BaseAgent):
    name = "VisualGenerationAgent"

    async def run(self, state):
        gen = get_image_generator()
        loop = asyncio.get_event_loop()

        char_seeds = {c.name: c.seed for c in state.characters}

        for scene in state.scenes:
            if not scene.image_prompt:
                self.logger.warning(f"Scene {scene.index} has no prompt, skipping.")
                continue

            # Prefer the seed of the protagonist or first character in the
            # scene; fall back to a stable per-scene seed.
            primary = next((c for c in scene.characters if c in char_seeds), None)
            if primary:
                seed = char_seeds[primary]
            else:
                seed = 42 + scene.index

            path = await loop.run_in_executor(
                None,
                lambda p=scene.image_prompt, s=seed: gen.generate(p, seed=s),
            )
            scene.image_path = str(path)
            self.logger.info(f"Scene {scene.index} image: {path.name} (seed={seed})")

        return state
