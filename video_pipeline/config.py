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


# ---- Style presets -------------------------------------------------------
# Each preset = (positive suffix, negative prompt, default SDXL model id).
# Pick via STYLE_PRESET env var. "cocomelon" is the default for kid-friendly
# 3D-cartoon story videos with consistent characters.
STYLE_PRESETS = {
    "cocomelon": {
        "suffix": (
            "3D pixar-style cartoon, cocomelon style, soft rounded shapes, "
            "vibrant saturated colors, big expressive eyes, friendly characters, "
            "clean smooth shading, bright cheerful daylight, plain uncluttered background, "
            "high detail, sharp focus, kid-friendly, octane render, ultra HD"
        ),
        "negative": (
            "photorealistic, gritty, dark, scary, blood, weapon, low quality, "
            "blurry, deformed, extra limbs, text, watermark, signature, "
            "ugly, distorted face, asymmetric eyes, low contrast, grainy"
        ),
        "model": "Lykon/dreamshaper-xl-v2-turbo",
        "character_template": (
            "3D pixar-style cartoon character, cocomelon style, large round head, "
            "big expressive eyes, soft rounded body, simple bright clothing, "
            "smooth toon shading, friendly smile"
        ),
    },
    "cinematic": {
        "suffix": (
            "cinematic, photorealistic, ultra-detailed, 8k, depth of field, "
            "shot on ARRI Alexa, 35mm anamorphic lens, film grain, "
            "professional color grading, volumetric lighting, sharp focus"
        ),
        "negative": "blurry, low quality, distorted, watermark, text, deformed",
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "character_template": (
            "photorealistic adult, cinematic portrait, natural skin tones, "
            "realistic proportions, detailed features"
        ),
    },
    "anime": {
        "suffix": (
            "anime style, cel shaded, vibrant colors, clean line art, "
            "studio ghibli inspired, expressive characters, sharp focus, ultra HD"
        ),
        "negative": "photorealistic, 3d render, blurry, low quality, deformed, watermark, text",
        "model": "Linaqruf/animagine-xl-3.1",
        "character_template": (
            "anime character, cel shaded, expressive eyes, clean line art, "
            "stylized proportions"
        ),
    },
}


@dataclass
class PipelineConfig:
    """Central configuration for the entire video generation pipeline."""

    # ---- LLM Provider ----
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    # Default to Sonnet 4.6 — Opus is overkill (and 5x cost) for this pipeline.
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    # Cheap model for prompt-eng / refinement; Haiku is ~10x cheaper than Sonnet.
    LLM_FAST_MODEL: str = os.getenv("LLM_FAST_MODEL", "claude-haiku-4-5-20251001")
    # Anthropic prompt caching — saves up to 90% on repeated system prompts.
    ENABLE_PROMPT_CACHE: bool = os.getenv("ENABLE_PROMPT_CACHE", "true").lower() == "true"

    # ---- Style ----
    STYLE_PRESET: str = os.getenv("STYLE_PRESET", "cocomelon")  # cocomelon | cinematic | anime

    # ---- Image / Video Generation ----
    # SD_MODEL_ID can be overridden; otherwise the preset's default model is used.
    SD_MODEL_ID: Optional[str] = os.getenv("SD_MODEL_ID") or None
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"
    # Low-VRAM mode for 8 GB GPUs (RTX 4060 / 3060 / 2080). Enables CPU offload,
    # VAE tiling, smaller resolution and fewer steps.
    LOW_VRAM_MODE: bool = os.getenv("LOW_VRAM_MODE", "true").lower() == "true"
    # 1280x720 fits comfortably in 8 GB; render is upscaled in moviepy if needed.
    IMAGE_WIDTH: int = int(os.getenv("IMAGE_WIDTH", "1280"))
    IMAGE_HEIGHT: int = int(os.getenv("IMAGE_HEIGHT", "720"))
    # DreamShaper-XL Turbo / SDXL-Turbo work in 4-8 steps; vanilla SDXL needs ~25.
    NUM_INFERENCE_STEPS: int = int(os.getenv("NUM_INFERENCE_STEPS", "8"))
    GUIDANCE_SCALE: float = float(os.getenv("GUIDANCE_SCALE", "2.0"))

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
    DEFAULT_MUSIC_VOLUME: float = 0.10  # quieter by default so voice cuts through

    # ---- Video Output ----
    VIDEO_FPS: int = 24
    VIDEO_BITRATE: str = "8000k"
    OUTPUT_DIR: Path = Path("outputs")
    CACHE_DIR: Path = Path("outputs/.cache")

    # ---- Pipeline behavior ----
    MAX_RETRIES: int = 3
    SCENE_DEFAULT_DURATION: float = 5.0  # seconds per scene if unspecified
    # Hard floor/ceiling so a stretched/short narration doesn't make a 1s flash
    # or a 30s drone.
    MIN_SCENE_DURATION: float = 3.0
    MAX_SCENE_DURATION: float = 15.0
    # Total target screen-time. ScenePlanner will aim for this.
    TARGET_DURATION_SECONDS: float = float(os.getenv("TARGET_DURATION_SECONDS", "300"))
    # Buffer added to TTS audio length when sizing the scene clip — prevents
    # the last syllable getting clipped by a hard cut.
    VOICE_END_PAD: float = 0.6
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
        # Resolve SD model from preset if not explicitly overridden.
        if not self.SD_MODEL_ID:
            self.SD_MODEL_ID = self.style().get(
                "model", "stabilityai/stable-diffusion-xl-base-1.0"
            )

    def style(self) -> dict:
        """Active style preset dict."""
        return STYLE_PRESETS.get(self.STYLE_PRESET, STYLE_PRESETS["cocomelon"])

    def validate(self) -> list[str]:
        """Return a list of warnings about missing optional configs."""
        warnings = []
        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            warnings.append("ANTHROPIC_API_KEY not set - LLM agents will fail.")
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            warnings.append("OPENAI_API_KEY not set - LLM agents will fail.")
        if self.TTS_PROVIDER == "elevenlabs" and not self.ELEVENLABS_API_KEY:
            warnings.append("ELEVENLABS_API_KEY not set - falling back to Coqui TTS.")
        if self.STYLE_PRESET not in STYLE_PRESETS:
            warnings.append(
                f"Unknown STYLE_PRESET={self.STYLE_PRESET}, defaulting to cocomelon."
            )
        return warnings


# Singleton instance
config = PipelineConfig()
