"""
Image generation tool using HuggingFace Diffusers (SDXL).
Lazy-loads the pipeline on first use to avoid GPU memory hit at import.
"""
import logging
import hashlib
from pathlib import Path
from typing import Optional
from config import config

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Wraps Stable Diffusion XL for cinematic image generation."""

    def __init__(self):
        self._pipe = None
        self._device = None

    def _load(self):
        """Load the diffusion pipeline on first use."""
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError:
            raise ImportError("Run: pip install torch diffusers transformers accelerate")

        self._device = "cuda" if (config.USE_GPU and torch.cuda.is_available()) else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        logger.info(f"Loading SDXL on {self._device}...")
        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            config.SD_MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if self._device == "cuda" else None,
        )
        self._pipe = self._pipe.to(self._device)
        if self._device == "cuda":
            self._pipe.enable_attention_slicing()
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        logger.info("SDXL loaded.")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, watermark, text, deformed",
        seed: Optional[int] = None,
        cache: bool = True,
    ) -> Path:
        """
        Generate one image and return the saved path.
        Uses prompt-hash caching to avoid regenerating identical prompts.
        """
        # Cache key
        key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        cache_path = config.CACHE_DIR / f"img_{key}.png"
        if cache and cache_path.exists():
            logger.info(f"Cache hit: {cache_path.name}")
            return cache_path

        self._load()
        import torch

        gen = torch.Generator(device=self._device)
        if seed is not None:
            gen.manual_seed(seed)

        logger.info(f"Generating image: {prompt[:80]}...")
        image = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=config.IMAGE_WIDTH,
            height=config.IMAGE_HEIGHT,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            generator=gen,
        ).images[0]

        image.save(cache_path)
        return cache_path


_singleton: Optional[ImageGenerator] = None

def get_image_generator() -> ImageGenerator:
    global _singleton
    if _singleton is None:
        _singleton = ImageGenerator()
    return _singleton
