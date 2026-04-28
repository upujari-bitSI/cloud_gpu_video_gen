"""
Rendering Agent: merges per-scene audio and video into final per-scene clips.
"""
import asyncio
from pathlib import Path
from agents.base import BaseAgent
from tools.video_utils import merge_audio_video
from config import config


class RenderingAgent(BaseAgent):
    name = "RenderingAgent"

    async def run(self, state):
        loop = asyncio.get_event_loop()
        out_dir = config.OUTPUT_DIR / "scenes"
        out_dir.mkdir(parents=True, exist_ok=True)

        for scene in state.scenes:
            if not scene.clip_path:
                self.logger.warning(f"Scene {scene.index}: no clip, skipping render.")
                continue
            out_path = out_dir / f"scene_{scene.index:03d}_final.mp4"

            if scene.voice_path and Path(scene.voice_path).exists():
                await loop.run_in_executor(
                    None,
                    merge_audio_video,
                    Path(scene.clip_path),
                    Path(scene.voice_path),
                    out_path,
                )
            else:
                # No voice: just copy the silent animation
                from shutil import copy
                copy(scene.clip_path, out_path)

            scene.final_clip_path = str(out_path)
            self.logger.info(f"Scene {scene.index} rendered -> {out_path.name}")
        return state
