"""
Animation Agent: turns each static image into a short animated clip
using Ken Burns / parallax-style motion.
"""
import asyncio
from pathlib import Path
from agents.base import BaseAgent
from tools.video_utils import animate_image
from config import config


# Map mood -> motion style for variety
MOOD_MOTION = {
    "tense": "zoom_in",
    "hopeful": "zoom_out",
    "melancholic": "pan_left",
    "exciting": "pan_right",
    "mysterious": "parallax",
}


class AnimationAgent(BaseAgent):
    name = "AnimationAgent"

    async def run(self, state):
        loop = asyncio.get_event_loop()
        clips_dir = config.OUTPUT_DIR / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        for scene in state.scenes:
            if not scene.image_path:
                self.logger.warning(f"Scene {scene.index}: no image, skipping animation.")
                continue
            motion = MOOD_MOTION.get(scene.mood.lower(), "zoom_in")
            out = clips_dir / f"scene_{scene.index:03d}_anim.mp4"
            await loop.run_in_executor(
                None,
                animate_image,
                Path(scene.image_path),
                scene.duration,
                out,
                motion,
            )
            scene.clip_path = str(out)
            self.logger.info(f"Scene {scene.index} animated -> {out.name} ({motion})")
        return state
