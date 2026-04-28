"""
Character Designer Agent: creates character specs with HUMAN-IN-THE-LOOP approval.
"""
from agents.base import BaseAgent
from tools.llm_client import get_llm
from state import Character
from config import config


class CharacterDesignerAgent(BaseAgent):
    name = "CharacterDesignerAgent"

    SYSTEM = """You design memorable, visually distinctive characters for cinematic shorts.
Each character must have a vivid description and a detailed visual prompt suitable for AI image generation."""

    USER_TPL = """Story:
Title: {title}
Synopsis: {synopsis}

Scene snippets to inform character design:
{scenes_summary}

Identify all named characters and design each one. Return JSON array:
[
  {{
    "name": "Character Name",
    "description": "narrative description (background, traits)",
    "visual_prompt": "detailed visual prompt: age, ethnicity, hair, clothing, distinctive features, body type, expressions. Use cinematic, photorealistic language.",
    "personality": "core personality traits",
    "role": "protagonist | antagonist | supporting"
  }}
]"""

    REFINE_TPL = """Original character:
{original}

User feedback: {feedback}

Update the character spec based on the feedback. Return the SAME JSON schema as before."""

    async def run(self, state):
        if not state.story or not state.scenes:
            raise ValueError("Need story and scenes before character design.")

        # Pull unique character names mentioned across scenes
        mentioned = set()
        for s in state.scenes:
            for c in s.characters:
                mentioned.add(c)

        scenes_summary = "\n".join(
            f"- {s.title}: {s.actions}" for s in state.scenes[:6]
        )

        llm = get_llm()
        chars_data = llm.complete_json(
            system=self.SYSTEM,
            user=self.USER_TPL.format(
                title=state.story.title,
                synopsis=state.story.synopsis,
                scenes_summary=scenes_summary,
            ),
        )
        if isinstance(chars_data, dict) and "characters" in chars_data:
            chars_data = chars_data["characters"]

        state.characters = [Character(**c) for c in chars_data]
        self.logger.info(f"Designed {len(state.characters)} characters.")

        # Human-in-the-loop approval
        if config.HUMAN_APPROVAL_REQUIRED:
            await self._approval_loop(state)

        return state

    async def _approval_loop(self, state):
        """Interactively show characters and accept feedback until approved."""
        while True:
            print("\n" + "=" * 60)
            print("CHARACTER DESIGNS - REVIEW")
            print("=" * 60)
            for i, c in enumerate(state.characters):
                print(f"\n[{i}] {c.name} ({c.role})")
                print(f"    Description: {c.description}")
                print(f"    Personality: {c.personality}")
                print(f"    Visual: {c.visual_prompt}")
            print("\n" + "=" * 60)
            print("Options:")
            print("  approve              - approve all and continue")
            print("  edit <index> <feedback>  - refine one character")
            print("  regenerate           - regenerate all from scratch")
            choice = input("\nYour choice: ").strip()

            if choice.lower() == "approve":
                for c in state.characters:
                    c.approved = True
                print("✓ Characters approved.")
                return
            elif choice.lower().startswith("edit "):
                parts = choice.split(maxsplit=2)
                if len(parts) < 3:
                    print("Usage: edit <index> <feedback>")
                    continue
                try:
                    idx = int(parts[1])
                    feedback = parts[2]
                    await self._refine_character(state, idx, feedback)
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}")
            elif choice.lower() == "regenerate":
                await self.run(state)
                return
            else:
                print("Unknown option.")

    async def _refine_character(self, state, idx: int, feedback: str):
        if idx < 0 or idx >= len(state.characters):
            print(f"Invalid index {idx}")
            return
        original = state.characters[idx]
        llm = get_llm()
        from dataclasses import asdict
        import json
        updated = llm.complete_json(
            system=self.SYSTEM,
            user=self.REFINE_TPL.format(
                original=json.dumps(asdict(original), indent=2),
                feedback=feedback,
            ),
        )
        if isinstance(updated, list):
            updated = updated[0]
        state.characters[idx] = Character(**{k: updated.get(k, getattr(original, k)) for k in
            ["name", "description", "visual_prompt", "personality", "role"]})
        print(f"✓ Updated character {state.characters[idx].name}")
