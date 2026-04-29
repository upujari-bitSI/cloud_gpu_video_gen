"""
Character Portrait Agent.

For each designed character, generates ONE reference portrait using the
character's visual_prompt + stable seed. Runs after CharacterDesigner and
before PromptEngineer / VisualGeneration so that:

  1. The dashboard can show character portraits live, before sinking minutes
     into 30+ scene generations with a wrong character look.
  2. Downstream IP-Adapter / PuLID face-locking (future phase) has a single
     reference image per character to condition on.

Same seed every run = stable, deterministic portraits. Failed portraits are
non-fatal — pipeline continues with portrait_path=None and falls back to the
text-only character spec for scene prompts.
"""
import asyncio
from pathlib import Path

from agents.base import BaseAgent
from tools.image_gen import get_image_generator
from config import config


class CharacterPortraitAgent(BaseAgent):
    name = "CharacterPortraitAgent"

    async def run(self, state):
        if not state.characters:
            self.logger.info("No characters to portrait — skipping.")
            return state

        loop = asyncio.get_event_loop()
        portrait_dir = config.OUTPUT_DIR / "characters"
        portrait_dir.mkdir(parents=True, exist_ok=True)

        gen = get_image_generator()
        style = config.style()
        style_suffix = style.get("suffix", "")

        for char in state.characters:
            if char.portrait_path and Path(char.portrait_path).exists():
                self.logger.info(f"{char.name}: portrait cached, skipping.")
                continue

            # Compose a clean portrait prompt: subject-only, neutral background,
            # so face/outfit features dominate over scene context.
            portrait_prompt = (
                f"Character portrait of {char.name}, {char.visual_prompt}, "
                f"centered headshot, plain neutral background, neutral expression, "
                f"front facing, soft even lighting, full character visible from waist up. "
                f"{style_suffix}"
            )

            try:
                path = await loop.run_in_executor(
                    None,
                    lambda p=portrait_prompt, s=char.seed: gen.generate(p, seed=s),
                )
                # Copy from the gen cache into a stable, named location for the
                # dashboard + future PuLID lookups.
                stable_path = portrait_dir / f"{_safe_name(char.name)}.png"
                if Path(path) != stable_path:
                    stable_path.write_bytes(Path(path).read_bytes())
                char.portrait_path = str(stable_path)
                self.logger.info(f"{char.name} portrait -> {stable_path.name}")
            except Exception as e:
                self.logger.warning(f"{char.name}: portrait generation failed ({e})")
                # Non-fatal; downstream agents handle portrait_path=None.
        return state


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)[:60] or "character"
