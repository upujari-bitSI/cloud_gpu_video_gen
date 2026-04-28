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
        # Pipeline order is intentional - dependencies enforced by data
        self.pipeline = [
            ("Story", StoryAgent()),
            ("ScenePlanner", ScenePlannerAgent()),
            ("CharacterDesigner", CharacterDesignerAgent()),  # human approval here
            ("PromptEngineer", PromptEngineerAgent()),
            ("VisualGeneration", VisualGenerationAgent()),
            ("Animation", AnimationAgent()),
            ("VoiceOver", VoiceOverAgent()),
            ("Music", MusicAgent()),
            ("Rendering", RenderingAgent()),
            ("FinalStitching", FinalStitchingAgent()),
        ]

    async def run(self, niche: str) -> PipelineState:
        """Run the full pipeline for a given niche."""
        state = PipelineState(niche=niche)
        self.logger.info(f"=== PIPELINE START: '{niche}' ===")

        # Validate config and surface warnings up front
        for w in config.validate():
            self.logger.warning(w)

        for label, agent in self.pipeline:
            self.logger.info(f"--- Running {label} ---")
            try:
                state = await agent.run_with_retry(state)
                # Persist intermediate state for resumability/debugging
                state.save(config.OUTPUT_DIR / "state.json")
            except Exception as e:
                self.logger.error(f"FATAL in {label}: {e}")
                state.errors.append(f"{label}: {e}")
                state.save(config.OUTPUT_DIR / "state_error.json")
                raise

        self.logger.info(f"=== PIPELINE COMPLETE ===")
        self.logger.info(f"Final video: {state.final_video_path}")
        return state
