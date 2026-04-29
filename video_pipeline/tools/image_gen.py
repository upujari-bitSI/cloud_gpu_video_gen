"""
Image generation tool. Supports two backends auto-selected by model id:

  * Flux.1 [schnell] / [dev]  - 12B-param transformer-based model with state-
    of-the-art prompt fidelity and text rendering. 4-step inference, no
    negative prompt, fp16/bf16 only. Default for new projects.
  * SDXL family (DreamShaper-XL Turbo, SDXL-Turbo, SDXL-Base, etc.) - 3.5B
    UNet, kept as a faster lightweight fallback for low-VRAM machines.

Pipeline output and the cache key format are identical across backends so
upstream agents (PromptEngineer, VisualGeneration) need no changes.

LOW_VRAM_MODE enables CPU offload + VAE slicing/tiling. Flux fits in
~16GB VRAM with offload; SDXL fits in ~6GB.
"""
import logging
import hashlib
from pathlib import Path
from typing import Optional
from config import config

logger = logging.getLogger(__name__)


def _is_flux(model_id: Optional[str]) -> bool:
    return "flux" in (model_id or "").lower()


class ImageGenerator:
    """Wraps a diffusers text-to-image pipeline (Flux or SDXL)."""

    def __init__(self):
        self._pipe = None
        self._device = None
        self._is_flux = False

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch
        except ImportError as e:
            raise ImportError(f"torch not installed: {e}") from e

        self._device = "cuda" if (config.USE_GPU and torch.cuda.is_available()) else "cpu"
        self._is_flux = _is_flux(config.SD_MODEL_ID)

        if self._is_flux:
            self._load_flux(torch)
        else:
            self._load_sdxl(torch)

    def _load_flux(self, torch):
        try:
            from diffusers import FluxPipeline
        except ImportError as e:
            raise ImportError(
                f"FluxPipeline missing: {e}. Need diffusers>=0.30 with Flux support. "
                "Run: pip install -U 'diffusers>=0.30,<0.32'"
            ) from e

        # Flux is trained in bf16; using fp16 introduces NaNs on long prompts.
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        logger.info(f"Loading Flux pipeline {config.SD_MODEL_ID} ({dtype}) on {self._device}...")
        self._pipe = FluxPipeline.from_pretrained(
            config.SD_MODEL_ID,
            torch_dtype=dtype,
        )

        if self._device == "cuda":
            # Flux always uses cpu_offload — even at 24GB it's tight without it.
            self._pipe.enable_model_cpu_offload()
            if hasattr(self._pipe, "enable_vae_slicing"):
                self._pipe.enable_vae_slicing()
            if hasattr(self._pipe, "enable_vae_tiling"):
                self._pipe.enable_vae_tiling()
            logger.info("Flux: cpu_offload + vae_slicing + vae_tiling")
        else:
            self._pipe = self._pipe.to(self._device)
        logger.info("Flux ready.")

    def _load_sdxl(self, torch):
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as e:
            raise ImportError(
                f"diffusers missing/incompatible: {e}\n"
                "Run: bash setup.sh"
            ) from e

        dtype = torch.float16 if self._device == "cuda" else torch.float32
        logger.info(f"Loading SDXL pipeline {config.SD_MODEL_ID} on {self._device}...")
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            config.SD_MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if self._device == "cuda" else None,
        )

        if self._device == "cuda":
            if config.LOW_VRAM_MODE:
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
        """Generate one image and return the saved path. Cached by inputs."""
        if negative_prompt is None:
            negative_prompt = config.style().get(
                "negative", "blurry, low quality, distorted, watermark, text, deformed"
            )

        # Cache key encodes model_id so swapping Flux<->SDXL invalidates correctly.
        key_src = f"{config.SD_MODEL_ID}|{prompt}|{negative_prompt}|{seed}"
        key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
        cache_path = config.CACHE_DIR / f"img_{key}.png"
        if cache and cache_path.exists():
            logger.info(f"Cache hit: {cache_path.name}")
            return cache_path

        self._load()
        import torch

        # Flux uses cpu_offload always, SDXL only when LOW_VRAM_MODE is on.
        cpu_offload = self._is_flux or config.LOW_VRAM_MODE
        gen_device = "cpu" if cpu_offload else self._device
        gen = torch.Generator(device=gen_device)
        if seed is not None:
            gen.manual_seed(seed)

        logger.info(f"Generating image (seed={seed}): {prompt[:80]}...")
        if self._is_flux:
            # Flux schnell sweet spot is 4 steps, guidance=0 (it's a guidance-
            # distilled model). Flux dev wants ~28 steps and guidance ~3.5.
            schnell = "schnell" in (config.SD_MODEL_ID or "").lower()
            steps = 4 if schnell else max(config.NUM_INFERENCE_STEPS, 20)
            guidance = 0.0 if schnell else max(config.GUIDANCE_SCALE, 3.5)
            image = self._pipe(
                prompt=prompt,
                width=config.IMAGE_WIDTH,
                height=config.IMAGE_HEIGHT,
                num_inference_steps=steps,
                guidance_scale=guidance,
                max_sequence_length=256,
                generator=gen,
            ).images[0]
        else:
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
