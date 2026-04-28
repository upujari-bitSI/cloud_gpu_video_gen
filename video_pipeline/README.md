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

**Hardware target:** any GPU with **≥16 GB VRAM** runs SDXL comfortably. RTX 4090 (24 GB), A100 (40/80 GB), L40 (40 GB), and RTX 3090 (24 GB) are all great. RTX 5090 is overkill. Anything below 16 GB needs SDXL-Turbo or a smaller model.

**Cheapest providers right now (April 2026):**

| Provider | RTX 4090 | A100 80 GB | H100 80 GB | Notes |
|----------|----------|------------|------------|-------|
| **Vast.ai** | ~$0.16–0.35/hr | ~$0.67/hr | ~$1.49–1.87/hr | Cheapest. Marketplace — instances can disappear. |
| **RunPod (Community)** | ~$0.34/hr | ~$0.79/hr | ~$1.99/hr | Best balance of cheap + reliable. Pre-built SDXL templates. |
| **Clore.ai / TensorDock** | ~$0.10–0.30/hr | ~$0.75/hr | ~$1.80/hr | Newer P2P platforms, very low fees. |
| **Lambda Labs** | n/a | $1.10/hr (40 GB) | $2.49–$2.99/hr | Zero egress fees — good if your output is large. |
| **Paperspace / Hyperstack** | mid-tier | mid-tier | mid-tier | Easier UX, slightly pricier. |
| AWS / GCP / Azure | — | $4+/hr | $3.90+/hr | Avoid unless you need their ecosystem. |

(Pricing snapshot from Vast.ai, RunPod, Spheron, IntuitionLabs, and Northflank market trackers as of April 2026 — rates fluctuate daily, always check live before booking.)

### My recommendation

**For this pipeline → RunPod Community Cloud, RTX 4090, ~$0.34/hr.** Here's why:

1. **One-click SDXL templates** — pick "Stable Diffusion" or "PyTorch 2.x" and CUDA + drivers are already there.
2. **Per-second billing** — pay only for the ~10–20 minutes a full pipeline run takes.
3. **24 GB VRAM** — plenty for SDXL 1080×1920 at 30 steps.
4. **Volume mounts** — keep your model cache between sessions (HuggingFace downloads are slow).

Estimated cost per video generated: **about $0.10–0.20** (10–30 minutes of GPU time per ~6-scene video).

### Quick start on RunPod

```bash
# 1. On RunPod: deploy a Pod
#    Template: "RunPod PyTorch 2.4"
#    GPU: RTX 4090 (24 GB) — Community Cloud
#    Volume: 50 GB persistent (for HF model cache)

# 2. SSH in or open the web terminal
git clone <your-repo> && cd video_pipeline

# 3. Install
apt-get update && apt-get install -y ffmpeg
pip install -r requirements.txt

# 4. Set env
export ANTHROPIC_API_KEY=sk-ant-...
export USE_GPU=true

# 5. Run
python main.py --niche "AI replacing jobs story" --no-approval
```

### Cost-saving tips

- **Use spot/interruptible instances** on Vast.ai or Spheron — 40-60% cheaper if your run is short and resumable.
- **Cache the SDXL model** to a persistent volume — first download is ~7 GB.
- **Lower resolution** during testing (`IMAGE_WIDTH=1024, IMAGE_HEIGHT=576` in config.py) to iterate faster.
- **The pipeline caches images and TTS by prompt hash** — re-running the same niche barely uses the GPU.
- Run with `--no-approval` for batch generation; use the Streamlit UI only when you actively want to refine characters.

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
