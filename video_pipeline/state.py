"""
Shared data models and pipeline state.
Used by all agents to pass structured data through the pipeline.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
import json


@dataclass
class Character:
    """A character in the story."""
    name: str
    description: str
    visual_prompt: str  # detailed prompt for image gen
    personality: str = ""
    role: str = ""  # protagonist, antagonist, supporting
    approved: bool = False
    # Stable per-character seed so SDXL renders the same face/outfit each scene.
    seed: int = 0


@dataclass
class Scene:
    """A single scene in the story."""
    index: int
    title: str
    narration: str  # voice-over text
    environment: str
    mood: str
    camera: str  # camera angle/movement
    lighting: str
    characters: list[str] = field(default_factory=list)  # character names present
    actions: str = ""
    duration: float = 5.0
    image_prompt: str = ""  # filled by Prompt Engineering Agent
    image_path: Optional[str] = None  # filled by Visual Generation Agent
    clip_path: Optional[str] = None  # filled by Animation Agent
    voice_path: Optional[str] = None  # filled by Voice Over Agent
    final_clip_path: Optional[str] = None  # filled by Rendering Agent


@dataclass
class Story:
    """The complete story structure."""
    niche: str
    title: str
    logline: str  # one-sentence summary
    synopsis: str
    acts: list[dict] = field(default_factory=list)  # [{title, summary}, ...]


@dataclass
class PipelineState:
    """Mutable shared state passed between agents."""
    niche: str
    story: Optional[Story] = None
    scenes: list[Scene] = field(default_factory=list)
    characters: list[Character] = field(default_factory=list)
    final_video_path: Optional[str] = None
    music_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    # Live progress fields written by the orchestrator before/after each stage,
    # consumed by the Streamlit dashboard for real-time updates.
    current_stage: Optional[str] = None  # name of stage currently running
    completed_stages: list[str] = field(default_factory=list)
    started_at: Optional[float] = None  # unix ts of pipeline start
    updated_at: Optional[float] = None  # unix ts of last state.save()

    def save(self, path: str | Path):
        """Persist state to JSON for resuming or debugging."""
        import time
        self.updated_at = time.time()
        data = {
            "niche": self.niche,
            "story": asdict(self.story) if self.story else None,
            "scenes": [asdict(s) for s in self.scenes],
            "characters": [asdict(c) for c in self.characters],
            "final_video_path": self.final_video_path,
            "music_path": self.music_path,
            "errors": self.errors,
            "current_stage": self.current_stage,
            "completed_stages": self.completed_stages,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PipelineState":
        data = json.loads(Path(path).read_text())
        state = cls(niche=data["niche"])
        if data.get("story"):
            state.story = Story(**data["story"])
        state.scenes = [Scene(**s) for s in data.get("scenes", [])]
        state.characters = [Character(**c) for c in data.get("characters", [])]
        state.final_video_path = data.get("final_video_path")
        state.music_path = data.get("music_path")
        state.errors = data.get("errors", [])
        state.current_stage = data.get("current_stage")
        state.completed_stages = data.get("completed_stages", [])
        state.started_at = data.get("started_at")
        state.updated_at = data.get("updated_at")
        return state
