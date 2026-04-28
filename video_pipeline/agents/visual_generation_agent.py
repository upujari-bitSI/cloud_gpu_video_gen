"""
Visual Generation Agent: generates one image per scene using SDXL.
"""
import asyncio
from pathlib import Path
from agents.base import BaseAgent
from tools.image_gen import get_image_generator


class VisualGenerationAgent(BaseAgent):
    name = "VisualGenerationAgent"

    async def run(self, state):
        gen = get_image_generator()
        loop = asyncio.get_event_loop()

        for scene in state.scenes:
            if not scene.image_prompt:
                self.logger.warning(f"Scene {scene.index} has no prompt, skipping.")
                continue
            # SDXL is sync and GPU-bound -- run in executor so async still works
            path = await loop.run_in_executor(
                None,
                lambda p=scene.image_prompt: gen.generate(p, seed=42 + scene.index),
            )
            scene.image_path = str(path)
            self.logger.info(f"Scene {scene.index} image: {path.name}")

        return state
