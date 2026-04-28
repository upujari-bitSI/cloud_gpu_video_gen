"""
Voice Over Agent: generates narration audio for each scene.
"""
import asyncio
from pathlib import Path
from agents.base import BaseAgent
from tools.tts import get_tts
from config import config


class VoiceOverAgent(BaseAgent):
    name = "VoiceOverAgent"

    async def run(self, state):
        tts = get_tts()
        loop = asyncio.get_event_loop()
        voice_dir = config.OUTPUT_DIR / "voice"
        voice_dir.mkdir(parents=True, exist_ok=True)

        for scene in state.scenes:
            if not scene.narration:
                continue
            out = voice_dir / f"scene_{scene.index:03d}_voice.mp3"
            path = await loop.run_in_executor(
                None,
                tts.synthesize,
                scene.narration,
                out,
                scene.mood,  # used for emotion
            )
            scene.voice_path = str(path)
            self.logger.info(f"Scene {scene.index} voice -> {Path(path).name}")
        return state
