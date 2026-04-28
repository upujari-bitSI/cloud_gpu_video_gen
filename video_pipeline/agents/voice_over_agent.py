"""
Voice Over Agent: generates narration audio for each scene.

Critical sync step: after each TTS call we measure the actual audio length
and write it back to scene.duration (clamped to [MIN, MAX]). The Animation
agent that runs after this then sizes its Ken Burns clip to match the voice
exactly, plus a small tail buffer so the last syllable never gets cut.
"""
import asyncio
from pathlib import Path
from agents.base import BaseAgent
from tools.tts import get_tts, get_audio_duration
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
                scene.mood,
            )
            scene.voice_path = str(path)

            # Resize the scene to the actual narration length + tail buffer.
            try:
                audio_seconds = get_audio_duration(Path(path))
                target = audio_seconds + config.VOICE_END_PAD
                target = max(config.MIN_SCENE_DURATION,
                             min(config.MAX_SCENE_DURATION, target))
                scene.duration = round(target, 2)
            except Exception as e:
                self.logger.warning(
                    f"Scene {scene.index}: could not measure audio duration ({e}); "
                    f"keeping planned duration {scene.duration}s"
                )

            self.logger.info(
                f"Scene {scene.index} voice -> {Path(path).name} "
                f"(duration set to {scene.duration}s)"
            )
        return state
