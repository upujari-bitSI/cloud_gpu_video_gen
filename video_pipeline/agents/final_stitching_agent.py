"""
Final Stitching Agent: concatenates all per-scene clips with transitions
and overlays background music. Produces the final MP4.
"""
import asyncio
from pathlib import Path
from agents.base import BaseAgent
from tools.video_utils import stitch_clips
from config import config


class FinalStitchingAgent(BaseAgent):
    name = "FinalStitchingAgent"

    async def run(self, state):
        clip_paths = [
            Path(s.final_clip_path) for s in state.scenes
            if s.final_clip_path and Path(s.final_clip_path).exists()
        ]
        if not clip_paths:
            raise RuntimeError("No rendered clips available to stitch.")

        out_dir = config.OUTPUT_DIR / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        title_safe = "".join(
            c if c.isalnum() else "_" for c in (state.story.title if state.story else "video")
        )[:50]
        out_path = out_dir / f"{title_safe}_final.mp4"

        loop = asyncio.get_event_loop()
        music = Path(state.music_path) if state.music_path else None
        await loop.run_in_executor(
            None,
            stitch_clips,
            clip_paths,
            out_path,
            0.5,  # transition duration
            music,
        )
        state.final_video_path = str(out_path)
        self.logger.info(f"✓ FINAL VIDEO: {out_path}")
        return state
