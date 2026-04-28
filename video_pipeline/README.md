# 🎬 AI Multi-Agent Video Generator

Multi-agent Python pipeline that turns a niche/topic into a full cinematic short:
**niche → story → scenes → characters → images → animation → voice → music → final MP4**

## Architecture

11 specialized agents coordinated by an Orchestrator:

| # | Agent | Job |
|---|-------|-----|
| 1 | Orchestrator | Drives flow, retries, state |
| 2 | Story | niche → 3-act story |
| 3 | ScenePlanner | story → detailed scenes |
| 4 | CharacterDesigner | characters + **human approval** |
| 5 | PromptEngineer | scenes → SDXL prompts |
| 6 | VisualGeneration | SDXL images |
| 7 | Animation | Ken Burns / parallax |
| 8 | VoiceOver | TTS narration |
| 9 | Music | Mood-matched BGM |
| 10 | Rendering | merge audio+video per scene |
| 11 | FinalStitching | concatenate + transitions + MP4 |

## Project Structure

```
video_pipeline/
├── agents/                 # 11 agent classes
│   ├── orchestrator.py
│   ├── story_agent.py
│   ├── scene_planner_agent.py
│   ├── character_designer_agent.py
│   ├── prompt_engineer_agent.py
│   ├── visual_generation_agent.py
│   ├── animation_agent.py
│   ├── voice_over_agent.py
│   ├── music_agent.py
│   ├── rendering_agent.py
│   ├── final_stitching_agent.py
│   └── base.py
├── tools/
│   ├── llm_client.py       # Claude/OpenAI wrapper
│   ├── image_gen.py        # SDXL via diffusers
│   ├── tts.py              # Coqui + ElevenLabs
│   └── video_utils.py      # moviepy helpers
├── workflows/              # (future: alt pipelines)
├── assets/
│   ├── music/              # drop royalty-free .mp3 here
│   └── characters/
├── outputs/
│   ├── scenes/             # per-scene clips with audio
│   ├── voice/              # voice-over audio
│   ├── clips/              # silent animated clips
│   ├── final/              # FINAL MP4
│   └── .cache/             # image/TTS cache
├── config.py               # central config
├── state.py                # data models
├── main.py                 # CLI entry
├── app.py                  # Streamlit UI (optional)
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Clone & enter
cd video_pipeline

# 2. Create venv
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Install ffmpeg (system-level)
# Ubuntu:  sudo apt install ffmpeg
# macOS:   brew install ffmpeg
# Windows: choco install ffmpeg

# 5. Configure
cp .env.example .env
# Edit .env -> add ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# 6. (Optional) drop a few royalty-free .mp3 files into assets/music/
#    Filenames matter: the music agent matches mood keywords like "tense", "hopeful", etc.
#    Free sources: pixabay.com/music, freesound.org, incompetech.com
```

## Usage

### CLI

```bash
python main.py --niche "AI replacing jobs story"
```

Skip the character approval step:

```bash
python main.py --niche "ancient Mars civilization" --no-approval
```

### Streamlit UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` — type a niche, review/refine characters in the browser, watch the final video render inline.

## Example Output

Input: `"AI replacing jobs story"`

Pipeline output:
- `outputs/final/Hollow_Office_final.mp4` — final 1080p MP4
- `outputs/state.json` — full pipeline state for inspection
- `outputs/pipeline.log` — full log

---

## ☁️ Cloud GPU — Where to Run This

This pipeline needs a GPU for SDXL image generation. CPU works but is ~50× slower.

The defaults (in `config.py`) are tuned for an **8 GB VRAM** GPU using
DreamShaper-XL Turbo (4–8 inference steps), CPU offload, and 1280×720 output.
Set `LOW_VRAM_MODE=false` and bump `IMAGE_WIDTH/HEIGHT` to 1920×1080 if you
have ≥16 GB.

### Cheapest providers (snapshot — always check live prices)

| Provider | RTX 3060 12 GB | RTX 4060 Ti 16 GB | RTX A4000 16 GB | RTX 4090 24 GB | Notes |
|---|---|---|---|---|---|
| **Vast.ai** | ~$0.08–0.15/hr | ~$0.18–0.25/hr | ~$0.20–0.30/hr | ~$0.20–0.40/hr | Cheapest, marketplace, instances can disappear |
| **RunPod (Community)** | n/a | n/a | ~$0.17–0.32/hr | ~$0.34–0.69/hr | Reliable, pre-built PyTorch/SD templates |
| **TensorDock** | ~$0.10/hr | ~$0.20/hr | ~$0.25/hr | ~$0.35/hr | Hourly + spot, no commitment |
| **Lambda Labs** | n/a | n/a | n/a | n/a | A10/A100 only, zero egress |

**Note on RTX 4060**: the consumer 4060 (8 GB) is rare on cloud GPU
marketplaces. The closest equivalents you'll actually find listed are:
- **RTX 3060 12 GB** on Vast.ai (~$0.10/hr) — perfect for this pipeline
- **RTX A4000 16 GB** on RunPod (~$0.20/hr) — most reliable cheap option
- **RTX 4060 Ti 16 GB** on Vast.ai (~$0.20/hr) — when available

### My recommendation: RunPod RTX A4000 (16 GB) at ~$0.20/hr

For this pipeline, **RunPod Community RTX A4000 (16 GB)** at ~$0.20/hr is the
sweet spot: enough VRAM to run SDXL at 1280×720 with no offload (faster
inference), pre-built PyTorch templates, persistent volumes for the SDXL
weights, per-second billing.

Cheaper alternatives if you want to squeeze more:
- **Vast.ai RTX 3060 12 GB at ~$0.10/hr** with `LOW_VRAM_MODE=true` (default).
  Slightly slower per image but ~half the cost. Spot-priced instances on
  Vast.ai dip to $0.06/hr.
- **TensorDock RTX A4000 spot** at ~$0.15/hr — second-cheapest reliable option.

Avoid AWS/GCP/Azure for this — they charge 5–10× more for equivalent GPUs.

### Estimated cost per 5-minute video (~30 scenes)

| GPU | Per-image | Per video (GPU) | Claude API | **Total** |
|---|---|---|---|---|
| RTX 3060 12 GB (LOW_VRAM) | ~6s | ~3 min → $0.005 | ~$0.15 | **~$0.16** |
| RTX A4000 16 GB | ~3s | ~1.5 min → $0.005 | ~$0.15 | **~$0.16** |
| RTX 4090 24 GB | ~1.5s | ~45s → $0.005 | ~$0.15 | **~$0.16** |

Claude API dominates the cost — most of the savings come from prompt caching
and using Sonnet/Haiku (already on by default). The GPU bill is rounding
error if you spin the pod down between runs.

### Quick start on RunPod

```bash
# 1. RunPod: deploy a Pod
#    Template: "RunPod PyTorch 2.4"
#    GPU: RTX A4000 (16 GB) — Community Cloud, ~$0.20/hr
#    Volume: 50 GB persistent (for HF model cache)

# 2. Web terminal:
git clone <your-repo> && cd video_pipeline
apt-get update && apt-get install -y ffmpeg
pip install -r requirements.txt

# 3. Env
export ANTHROPIC_API_KEY=sk-ant-...
export USE_GPU=true
export STYLE_PRESET=cocomelon            # cocomelon | cinematic | anime
export TARGET_DURATION_SECONDS=300       # 5-minute video
export LOW_VRAM_MODE=true                # leave true for ≤16 GB GPUs
# export LLM_MODEL=claude-sonnet-4-6     # default; opus is overkill
# export LLM_FAST_MODEL=claude-haiku-4-5-20251001   # for prompt-eng

# 4. Run
python main.py --niche "a curious bunny who learns to share" --no-approval
```

### Cost-saving tips

- **Spot / interruptible instances** on Vast.ai are 40–60% cheaper. The
  pipeline caches images and TTS by prompt hash so an interrupted run
  resumes near-free.
- **Cache the SDXL model** to a persistent volume — DreamShaper-XL Turbo
  is ~6 GB on first download.
- **Anthropic prompt caching is on by default** (`ENABLE_PROMPT_CACHE=true`)
  — saves ~90% on input tokens for repeated agent system prompts.
- **PromptEngineer batches all scenes into one Haiku call** — 1 API call
  per video instead of one per scene.
- **Iterate at low res**: `IMAGE_WIDTH=768 IMAGE_HEIGHT=432` for fast drafts.
- Use `--no-approval` for batch runs; use the Streamlit UI only when you
  want to refine characters.

---

## Customization

- **Different LLM:** set `LLM_PROVIDER=openai` and `LLM_MODEL=gpt-4o` in `.env`.
- **Better voice:** set `TTS_PROVIDER=elevenlabs` + your API key.
- **Different image model:** change `SD_MODEL_ID` to any HF Diffusers-compatible model (e.g., Playground, RealVis, Juggernaut XL).
- **Real video models:** swap `Animation Agent` to call Runway Gen-3 or Pika 1.5 if you have those API keys.

## Troubleshooting

- **CUDA OOM:** lower `IMAGE_WIDTH`/`IMAGE_HEIGHT` in `config.py` or use `enable_model_cpu_offload()` in `tools/image_gen.py`.
- **MoviePy error:** make sure ffmpeg is on PATH (`ffmpeg -version`).
- **Coqui TTS first-run is slow:** it downloads a model (~150 MB).
- **HuggingFace rate limit:** `huggingface-cli login` with a token.

## License

MIT — do whatever you want with it.
