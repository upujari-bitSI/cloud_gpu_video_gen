"""
Streamlit dashboard for the AI Video Pipeline.

Run from the video_pipeline/ directory:
    streamlit run streamlit_app.py

The dashboard launches main.py as a subprocess and live-tails outputs/state.json
to show:
  - overall stage progress (10 stages)
  - story title + synopsis
  - per-character visual specs and seeds
  - per-scene cards with thumbnail, narration, audio playback, animation, final
  - tail of the pipeline.log
  - the final stitched video at the end

No new agent code runs in this process — the dashboard is a passive observer
of state.json + files written under outputs/. That keeps the dashboard
crash-resistant: a pipeline failure just shows up as an error panel, the UI
keeps working.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
STATE_PATH = OUTPUT_DIR / "state.json"
LOG_PATH = OUTPUT_DIR / "pipeline.log"

STAGES = [
    "Story",
    "ScenePlanner",
    "CharacterDesigner",
    "CharacterPortrait",
    "PromptEngineer",
    "VisualGeneration",
    "VoiceOver",
    "Animation",
    "Music",
    "Rendering",
    "FinalStitching",
]

st.set_page_config(page_title="AI Video Pipeline", layout="wide", page_icon="🎬")


# ---------- helpers --------------------------------------------------------

def load_state() -> dict | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        # Mid-write race; treat as transient.
        return None


def tail_log(n: int = 60) -> str:
    if not LOG_PATH.exists():
        return ""
    try:
        lines = LOG_PATH.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def proc_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def launch_pipeline(niche: str, resume: bool) -> int:
    cmd = [sys.executable, "main.py", "--niche", niche, "--no-approval"]
    if resume:
        cmd.append("--resume")
    # Detach so closing the browser tab doesn't kill the run.
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


def stop_pipeline(pid: int):
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------- session state --------------------------------------------------

if "pipeline_pid" not in st.session_state:
    st.session_state.pipeline_pid = None
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True


# ---------- sidebar: controls ---------------------------------------------

with st.sidebar:
    st.header("🎬 Pipeline")
    niche = st.text_input(
        "Story niche",
        value="Cinderella story disney in indian style",
        help="The topic/genre for the LLM to write a story about.",
    )

    pid = st.session_state.pipeline_pid
    running = proc_alive(pid)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶️ Run", disabled=running, use_container_width=True):
            st.session_state.pipeline_pid = launch_pipeline(niche, resume=False)
            st.rerun()
    with col_b:
        if st.button("⏯️ Resume", disabled=running, use_container_width=True):
            st.session_state.pipeline_pid = launch_pipeline(niche, resume=True)
            st.rerun()

    if running:
        st.success(f"Running (pid {pid})")
        if st.button("🛑 Stop", use_container_width=True):
            stop_pipeline(pid)
            st.session_state.pipeline_pid = None
            st.rerun()
    else:
        st.info("Idle")

    st.session_state.auto_refresh = st.checkbox(
        "Auto-refresh (2s)", value=st.session_state.auto_refresh
    )

    st.divider()
    st.caption(f"Project root: {ROOT}")
    st.caption(f"State: {STATE_PATH}")


# ---------- main: progress + scenes ---------------------------------------

state = load_state()

if state is None:
    st.title("🎬 AI Video Pipeline")
    st.info("No pipeline state yet. Enter a niche in the sidebar and click **Run**.")
    st.stop()

# Header
st.title(state.get("story", {}).get("title") or "🎬 AI Video Pipeline")
if state.get("story"):
    st.caption(state["story"].get("logline") or "")

# Progress bar
completed = set(state.get("completed_stages") or [])
current = state.get("current_stage")
progress_count = len(completed)
st.progress(progress_count / len(STAGES), text=f"{progress_count} / {len(STAGES)} stages complete")

# Stage chips
chip_cols = st.columns(len(STAGES))
for i, stage in enumerate(STAGES):
    with chip_cols[i]:
        if stage in completed:
            st.markdown(f"<div style='text-align:center;color:#22c55e;font-size:0.8em'>✅<br>{stage}</div>", unsafe_allow_html=True)
        elif stage == current:
            st.markdown(f"<div style='text-align:center;color:#3b82f6;font-size:0.8em'>⏳<br>**{stage}**</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:center;color:#6b7280;font-size:0.8em'>⬜<br>{stage}</div>", unsafe_allow_html=True)

# Timing
started = state.get("started_at")
updated = state.get("updated_at")
if started and updated:
    elapsed = updated - started
    st.caption(f"Elapsed: {fmt_duration(elapsed)}")

# Errors
for err in state.get("errors", []):
    st.error(err)

st.divider()

# Final video (if ready)
final = state.get("final_video_path")
if final and Path(ROOT / final).exists():
    st.subheader("🎉 Final video")
    st.video(str(ROOT / final))

# Tabs
tab_scenes, tab_chars, tab_story, tab_log = st.tabs(["Scenes", "Characters", "Story", "Log"])

with tab_story:
    story = state.get("story")
    if story:
        st.markdown(f"### {story.get('title','')}")
        st.markdown(f"**Logline:** {story.get('logline','')}")
        st.markdown("**Synopsis:**")
        st.write(story.get("synopsis",""))
        acts = story.get("acts") or []
        if acts:
            st.markdown("**Acts:**")
            for i, act in enumerate(acts, 1):
                st.markdown(f"- **Act {i}: {act.get('title','')}** — {act.get('summary','')}")
    else:
        st.info("Story not yet generated.")

with tab_chars:
    chars = state.get("characters") or []
    if not chars:
        st.info("No characters yet.")
    for c in chars:
        with st.expander(
            f"{c.get('name','?')} — {c.get('role','')}", expanded=True
        ):
            cols = st.columns([1, 2])
            with cols[0]:
                portrait = c.get("portrait_path")
                if portrait and Path(ROOT / portrait).exists():
                    st.image(str(ROOT / portrait), use_container_width=True)
                else:
                    st.caption("(portrait pending)")
            with cols[1]:
                st.markdown(f"**Description:** {c.get('description','')}")
                st.markdown(f"**Personality:** {c.get('personality','')}")
                st.markdown(f"**Visual prompt:** `{c.get('visual_prompt','')}`")
                st.caption(f"Seed: {c.get('seed','-')}")

with tab_scenes:
    scenes = state.get("scenes") or []
    if not scenes:
        st.info("No scenes yet.")
    for scene in scenes:
        idx = scene.get("index", "?")
        with st.expander(
            f"Scene {idx}: {scene.get('title','')} "
            f"({scene.get('duration','?')}s, mood: {scene.get('mood','')})",
            expanded=False,
        ):
            cols = st.columns([2, 3])
            with cols[0]:
                img = scene.get("image_path")
                if img and Path(ROOT / img).exists():
                    st.image(str(ROOT / img), use_container_width=True)
                else:
                    st.caption("(image not yet generated)")
                clip = scene.get("final_clip_path") or scene.get("clip_path")
                if clip and Path(ROOT / clip).exists():
                    st.video(str(ROOT / clip))
            with cols[1]:
                st.markdown(f"**Narration:** {scene.get('narration','')}")
                st.caption(f"Camera: {scene.get('camera','')} · Lighting: {scene.get('lighting','')}")
                if scene.get("characters"):
                    st.caption(f"Characters: {', '.join(scene['characters'])}")
                voice = scene.get("voice_path")
                if voice and Path(ROOT / voice).exists():
                    st.audio(str(ROOT / voice))
                if scene.get("image_prompt"):
                    st.markdown("**Image prompt:**")
                    st.code(scene["image_prompt"], language=None)

with tab_log:
    st.code(tail_log(80) or "(empty)", language=None)


# ---------- auto-refresh ---------------------------------------------------

if st.session_state.auto_refresh and (running or current):
    time.sleep(2)
    st.rerun()
