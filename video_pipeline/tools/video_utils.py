"""
Video utilities: animate static images, merge audio+video, stitch clips.
Uses MoviePy.
"""
import logging
from pathlib import Path
from typing import Optional
from config import config

logger = logging.getLogger(__name__)


def animate_image(
    image_path: Path,
    duration: float,
    output_path: Path,
    motion: str = "zoom_in",
) -> Path:
    """
    Convert a static image into a short clip with camera motion (Ken Burns).
    motion options: zoom_in, zoom_out, pan_left, pan_right, parallax.
    """
    from moviepy.editor import ImageClip, vfx

    clip = ImageClip(str(image_path)).set_duration(duration)
    clip = clip.set_fps(config.VIDEO_FPS)

    if motion == "zoom_in":
        clip = clip.resize(lambda t: 1 + 0.04 * t)
    elif motion == "zoom_out":
        clip = clip.resize(lambda t: 1.2 - 0.04 * t)
    elif motion == "pan_left":
        w = clip.w
        clip = clip.set_position(lambda t: (-20 * t, 0))
        clip = clip.resize(1.1)
    elif motion == "pan_right":
        clip = clip.set_position(lambda t: (20 * t, 0))
        clip = clip.resize(1.1)
    elif motion == "parallax":
        # Subtle zoom + slight pan combined
        clip = clip.resize(lambda t: 1 + 0.03 * t)
        clip = clip.set_position(lambda t: (5 * t, -3 * t))

    clip = clip.set_duration(duration)
    clip.write_videofile(
        str(output_path),
        fps=config.VIDEO_FPS,
        codec="libx264",
        audio=False,
        preset="medium",
        bitrate=config.VIDEO_BITRATE,
        logger=None,
    )
    clip.close()
    return output_path


def merge_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """Combine a video clip with a voice-over audio file."""
    from moviepy.editor import VideoFileClip, AudioFileClip

    video = VideoFileClip(str(video_path))
    audio = AudioFileClip(str(audio_path))

    # Trim the longer one to match the shorter
    duration = min(video.duration, audio.duration)
    video = video.subclip(0, duration)
    audio = audio.subclip(0, duration)

    final = video.set_audio(audio)
    final.write_videofile(
        str(output_path),
        fps=config.VIDEO_FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate=config.VIDEO_BITRATE,
        logger=None,
    )
    final.close()
    video.close()
    audio.close()
    return output_path


def stitch_clips(
    clip_paths: list[Path],
    output_path: Path,
    transition_duration: float = 0.5,
    music_path: Optional[Path] = None,
) -> Path:
    """Concatenate multiple clips with crossfade transitions and optional bg music."""
    from moviepy.editor import (
        VideoFileClip, concatenate_videoclips,
        AudioFileClip, CompositeAudioClip,
    )

    clips = []
    for p in clip_paths:
        c = VideoFileClip(str(p))
        clips.append(c)

    # Apply crossfade transition between adjacent clips
    if transition_duration > 0 and len(clips) > 1:
        for i in range(1, len(clips)):
            clips[i] = clips[i].crossfadein(transition_duration)
        final = concatenate_videoclips(clips, method="compose", padding=-transition_duration)
    else:
        final = concatenate_videoclips(clips, method="compose")

    # Mix background music if provided
    if music_path and Path(music_path).exists():
        try:
            music = AudioFileClip(str(music_path)).volumex(config.DEFAULT_MUSIC_VOLUME)
            # Loop or trim music to match video length
            if music.duration < final.duration:
                from moviepy.audio.fx.all import audio_loop
                music = audio_loop(music, duration=final.duration)
            else:
                music = music.subclip(0, final.duration)
            if final.audio is not None:
                mixed = CompositeAudioClip([final.audio, music])
            else:
                mixed = music
            final = final.set_audio(mixed)
        except Exception as e:
            logger.warning(f"Could not mix music: {e}")

    final.write_videofile(
        str(output_path),
        fps=config.VIDEO_FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate=config.VIDEO_BITRATE,
        logger=None,
    )
    final.close()
    for c in clips:
        c.close()
    return output_path
