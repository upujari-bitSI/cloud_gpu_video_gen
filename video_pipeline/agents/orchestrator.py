"""
Orchestrator Agent: drives the full pipeline end-to-end.
Manages state, retries, and the human-approval checkpoint.
"""
import logging
from agents.base import BaseAgent
from agents.story_agent import StoryAgent
from agents.scene_planner_agent import ScenePlannerAgent
from agents.character_designer_agent import CharacterDesignerAgent
from agents.prompt_engineer_agent import PromptEngineerAgent
from agents.visual_generation_agent import VisualGenerationAgent
from agents.animation_agent import AnimationAgent
from agents.voice_over_agent import VoiceOverAgent
from agents.music_agent import MusicAgent
from agents.rendering_agent import RenderingAgent
from agents.final_stitching_agent import FinalStitchingAgent
from state import PipelineState
from config import config


class Orchestrator(BaseAgent):
    name = "Orchestrator"

    def __init__(self):
        super().__init__()
        # Pipeline order is intentional. VoiceOver runs BEFORE Animation so
        # the per-scene Ken Burns clip is sized to the actual narration audio
        # length (VoiceOver writes the measured duration back into Scene),
        # which is what gives us tight voice/visual sync at render time.
        self.pipeline = [
            ("Story", StoryAgent()),
            ("ScenePlanner", ScenePlannerAgent()),
            ("CharacterDesigner", CharacterDesignerAgent()),  # human approval here
            ("PromptEngineer", PromptEngineerAgent()),
            ("VisualGeneration", VisualGenerationAgent()),
            ("VoiceOver", VoiceOverAgent()),
            ("Animation", AnimationAgent()),
            ("Music", MusicAgent()),
            ("Rendering", RenderingAgent()),
            ("FinalStitching", FinalStitchingAgent()),
        ]

    async def run(self, niche: str, resume: bool = False) -> PipelineState:
        """Run the full pipeline for a given niche."""
        state_path = config.OUTPUT_DIR / "state.json"
        if resume and state_path.exists():
            state = PipelineState.load(state_path)
            self.logger.info(f"=== PIPELINE RESUME: '{state.niche}' ===")
        else:
            state = PipelineState(niche=niche)
            self.logger.info(f"=== PIPELINE START: '{niche}' ===")

        # Validate config and surface warnings up front
        for w in config.validate():
            self.logger.warning(w)

        for label, agent in self.pipeline:
            if resume and self._is_stage_complete(label, state):
                self.logger.info(f"--- Skipping {label} (already complete) ---")
                continue
            self.logger.info(f"--- Running {label} ---")
            try:
                state = await agent.run_with_retry(state)
                # Persist intermediate state for resumability/debugging
                state.save(state_path)
            except Exception as e:
                self.logger.error(f"FATAL in {label}: {e}")
                state.errors.append(f"{label}: {e}")
                state.save(config.OUTPUT_DIR / "state_error.json")
                raise

        self.logger.info(f"=== PIPELINE COMPLETE ===")
        self.logger.info(f"Final video: {state.final_video_path}")
        return state

    @staticmethod
    def _is_stage_complete(label: str, state: PipelineState) -> bool:
        from pathlib import Path

        def file_ok(p):
            return p and Path(p).exists() and Path(p).stat().st_size > 0

        scenes = state.scenes
        if label == "Story":
            return state.story is not None
        if label == "ScenePlanner":
            return bool(scenes)
        if label == "CharacterDesigner":
            return bool(state.characters)
        if label == "PromptEngineer":
            return bool(scenes) and all(s.image_prompt for s in scenes)
        if label == "VisualGeneration":
            return bool(scenes) and all(file_ok(s.image_path) for s in scenes)
        if label == "VoiceOver":
            return bool(scenes) and all(file_ok(s.voice_path) for s in scenes)
        if label == "Animation":
            return bool(scenes) and all(file_ok(s.clip_path) for s in scenes)
        if label == "Music":
            return state.music_path is not None
        if label == "Rendering":
            return bool(scenes) and all(file_ok(s.final_clip_path) for s in scenes)
        if label == "FinalStitching":
            return file_ok(state.final_video_path)
        return False
