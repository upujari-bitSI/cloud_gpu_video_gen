"""
Optional Streamlit UI for the video pipeline.
Run with: streamlit run app.py
"""
import asyncio
import streamlit as st
from pathlib import Path
from agents.orchestrator import Orchestrator
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
from state import PipelineState, Character
from config import config


st.set_page_config(page_title="AI Video Studio", page_icon="🎬", layout="wide")
st.title("🎬 AI Multi-Agent Video Generator")

if "state" not in st.session_state:
    st.session_state.state = None
if "step" not in st.session_state:
    st.session_state.step = "input"


def run_async(coro):
    """Helper to run async code in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ------ STEP: Input ------
if st.session_state.step == "input":
    st.subheader("1. Enter your story niche")
    niche = st.text_input("Niche/topic", placeholder="e.g., AI replacing jobs story")
    if st.button("Generate Story", type="primary") and niche.strip():
        with st.spinner("Crafting story and scenes..."):
            state = PipelineState(niche=niche)
            state = run_async(StoryAgent().run(state))
            state = run_async(ScenePlannerAgent().run(state))
            # Run character designer WITHOUT terminal approval - we approve in UI
            config.HUMAN_APPROVAL_REQUIRED = False
            state = run_async(CharacterDesignerAgent().run(state))
            st.session_state.state = state
            st.session_state.step = "review"
            st.rerun()


# ------ STEP: Review ------
elif st.session_state.step == "review":
    state = st.session_state.state
    st.subheader(f"📖 {state.story.title}")
    st.write(state.story.synopsis)

    with st.expander("Scenes", expanded=False):
        for s in state.scenes:
            st.markdown(f"**Scene {s.index+1}: {s.title}** ({s.duration}s)")
            st.caption(f"{s.environment} | {s.mood} | {s.lighting}")
            st.write(s.narration)

    st.subheader("👥 Characters - Review & Approve")
    for i, c in enumerate(state.characters):
        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"### {c.name} ({c.role})")
                st.write(f"**Description:** {c.description}")
                st.write(f"**Personality:** {c.personality}")
                st.code(c.visual_prompt, language="text")
            with cols[1]:
                feedback = st.text_area(
                    "Feedback (to refine)", key=f"fb_{i}",
                    placeholder="e.g., make older, add scar..."
                )
                if st.button("Refine", key=f"ref_{i}") and feedback.strip():
                    with st.spinner(f"Refining {c.name}..."):
                        run_async(
                            CharacterDesignerAgent()._refine_character(state, i, feedback)
                        )
                        st.rerun()

    if st.button("✓ Approve All & Generate Video", type="primary"):
        for c in state.characters:
            c.approved = True
        st.session_state.step = "render"
        st.rerun()


# ------ STEP: Render ------
elif st.session_state.step == "render":
    state = st.session_state.state
    st.subheader("🎞 Generating video...")
    progress = st.progress(0)
    status = st.empty()

    pipeline = [
        ("Engineering prompts", PromptEngineerAgent()),
        ("Generating images", VisualGenerationAgent()),
        ("Animating scenes", AnimationAgent()),
        ("Recording voice-over", VoiceOverAgent()),
        ("Selecting music", MusicAgent()),
        ("Rendering scenes", RenderingAgent()),
        ("Stitching final video", FinalStitchingAgent()),
    ]
    total = len(pipeline)
    for i, (label, agent) in enumerate(pipeline):
        status.text(f"[{i+1}/{total}] {label}...")
        state = run_async(agent.run(state))
        progress.progress((i + 1) / total)

    st.session_state.state = state
    st.session_state.step = "done"
    st.rerun()


# ------ STEP: Done ------
elif st.session_state.step == "done":
    state = st.session_state.state
    st.success("🎉 Video generated!")
    if state.final_video_path and Path(state.final_video_path).exists():
        st.video(state.final_video_path)
        with open(state.final_video_path, "rb") as f:
            st.download_button("Download MP4", f, file_name=Path(state.final_video_path).name)
    if st.button("Start over"):
        st.session_state.state = None
        st.session_state.step = "input"
        st.rerun()
