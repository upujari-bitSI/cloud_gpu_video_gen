"""
Configuration module for the AI Video Generation Pipeline.
Centralizes all API keys, model settings, and pipeline parameters.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class PipelineConfig:
    """Central configuration for the entire video generation pipeline."""

    # ---- LLM Provider ----
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-opus-4-7")

    # ---- Image / Video Generation ----
    SD_MODEL_ID: str = os.getenv("SD_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"
    IMAGE_WIDTH: int = 1920
    IMAGE_HEIGHT: int = 1080
    NUM_INFERENCE_STEPS: int = 30
    GUIDANCE_SCALE: float = 7.5

    # Optional video model APIs
    RUNWAY_API_KEY: Optional[str] = os.getenv("RUNWAY_API_KEY")
    PIKA_API_KEY: Optional[str] = os.getenv("PIKA_API_KEY")

    # ---- TTS / Voice ----
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "coqui")  # "elevenlabs" or "coqui"
    ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    COQUI_MODEL: str = "tts_models/en/ljspeech/tacotron2-DDC"

    # ---- Music ----
    MUSIC_LIBRARY_DIR: Path = Path("assets/music")  # royalty-free .mp3 files
    DEFAULT_MUSIC_VOLUME: float = 0.15  # 0-1, lower = quieter under voice

    # ---- Video Output ----
    VIDEO_FPS: int = 24
    VIDEO_BITRATE: str = "8000k"
    OUTPUT_DIR: Path = Path("outputs")
    CACHE_DIR: Path = Path("outputs/.cache")

    # ---- Pipeline behavior ----
    MAX_RETRIES: int = 3
    SCENE_DEFAULT_DURATION: float = 5.0  # seconds per scene if unspecified
    HUMAN_APPROVAL_REQUIRED: bool = True
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ---- Folders auto-created ----
    PROJECT_FOLDERS: list = field(default_factory=lambda: [
        "outputs", "outputs/.cache", "outputs/scenes",
        "outputs/voice", "outputs/clips", "outputs/final",
        "assets/music", "assets/characters",
    ])

    def __post_init__(self):
        for folder in self.PROJECT_FOLDERS:
            Path(folder).mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Return a list of warnings about missing optional configs."""
        warnings = []
        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            warnings.append("ANTHROPIC_API_KEY not set - LLM agents will fail.")
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            warnings.append("OPENAI_API_KEY not set - LLM agents will fail.")
        if self.TTS_PROVIDER == "elevenlabs" and not self.ELEVENLABS_API_KEY:
            warnings.append("ELEVENLABS_API_KEY not set - falling back to Coqui TTS.")
        return warnings


# Singleton instance
config = PipelineConfig()
