"""
Image generation tool using HuggingFace Diffusers (SDXL).
Lazy-loads the pipeline on first use to avoid GPU memory hit at import.

LOW_VRAM_MODE enables CPU offload + VAE slicing/tiling so SDXL fits in
~6 GB of VRAM (RTX 4060/3060/2080). Combined with DreamShaper-XL Turbo
or SDXL-Turbo, image generation runs in 4-8 inference steps.
"""
import logging
import hashlib
from pathlib import Path
from typing import Optional
from config import config

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Wraps Stable Diffusion XL for stylized image generation."""

    def __init__(self):
        self._pipe = None
        self._device = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as e:
            raise ImportError(
                f"Missing or incompatible dependency: {e}\n"
                "Run: pip install 'torch>=2.1.0' 'diffusers>=0.27.0' 'transformers>=4.40.0' 'accelerate>=0.30.0'\n"
                "If diffusers is already installed, a version conflict with huggingface_hub may exist — "
                "run: pip install --upgrade 'diffusers>=0.27.0' 'huggingface_hub>=0.23.0'"
            ) from e

        self._device = "cuda" if (config.USE_GPU and torch.cuda.is_available()) else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        logger.info(f"Loading SDXL pipeline {config.SD_MODEL_ID} on {self._device}...")
        # AutoPipeline picks the correct class for SDXL/turbo/dreamshaper variants.
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            config.SD_MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if self._device == "cuda" else None,
        )

        if self._device == "cuda":
            if config.LOW_VRAM_MODE:
                # CPU offload swaps weights between RAM and VRAM as needed —
                # ~2x slower than .to('cuda') but fits 8 GB easily.
                self._pipe.enable_model_cpu_offload()
                self._pipe.enable_vae_slicing()
                self._pipe.enable_vae_tiling()
                logger.info("Low-VRAM mode: cpu_offload + vae_slicing + vae_tiling")
            else:
                self._pipe = self._pipe.to(self._device)
                self._pipe.enable_attention_slicing()
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        else:
            self._pipe = self._pipe.to(self._device)
        logger.info("SDXL ready.")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        cache: bool = True,
    ) -> Path:
        """
        Generate one image and return the saved path.
        Cache key = prompt + negative + seed, so reruns with the same inputs
        skip the GPU entirely.
        """
        if negative_prompt is None:
            negative_prompt = config.style().get(
                "negative", "blurry, low quality, distorted, watermark, text, deformed"
            )

        key_src = f"{prompt}|{negative_prompt}|{seed}"
        key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
        cache_path = config.CACHE_DIR / f"img_{key}.png"
        if cache and cache_path.exists():
            logger.info(f"Cache hit: {cache_path.name}")
            return cache_path

        self._load()
        import torch

        # Generator must live on CPU when CPU offload is on; otherwise on cuda.
        gen_device = "cpu" if config.LOW_VRAM_MODE else self._device
        gen = torch.Generator(device=gen_device)
        if seed is not None:
            gen.manual_seed(seed)

        logger.info(f"Generating image (seed={seed}): {prompt[:80]}...")
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
