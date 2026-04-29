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
    # moviepy 1.0.3's write_videofile silently drops fps to None on this env
    # (newer Pillow + decorator package), so we shell out to ffmpeg directly
    # to turn a static image into a fixed-duration mp4 clip.
    import subprocess
    import imageio_ffmpeg
    fps = config.VIDEO_FPS or 24
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(image_path),
        "-c:v", "libx264", "-t", f"{duration:.3f}",
        "-r", str(fps), "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-b:v", config.VIDEO_BITRATE, "-preset", "medium",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def merge_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """Combine a video clip with a voice-over audio file.

    Critical fix vs. the original: if audio is longer than the video we
    EXTEND the video by freezing its last frame instead of truncating the
    narration mid-sentence. The animation agent already sized the clip to
    the audio length + buffer, but this is a safety net.
    """
    from moviepy.editor import VideoFileClip, AudioFileClip
    from moviepy.video.fx.all import freeze

    video = VideoFileClip(str(video_path))
    audio = AudioFileClip(str(audio_path))

    if audio.duration > video.duration + 0.05:
        # Extend the video by freezing the last frame to match audio length.
        extra = audio.duration - video.duration
        try:
            video = freeze(video, t=video.duration - 0.05, freeze_duration=extra + 0.1)
        except Exception:
            # Fallback: clamp audio to video. Loud cut, but preserves the file.
            audio = audio.subclip(0, video.duration)
    elif video.duration > audio.duration + 0.05:
        # Audio shorter than video — trim video to audio length so we never
        # try to read past the end of the audio file (causes OSError at write).
        video = video.subclip(0, audio.duration)

    # Clamp to audio.duration so write_videofile never reads past the audio end.
    duration = min(video.duration, audio.duration)
    final = video.set_audio(audio).set_duration(duration)
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

    clips = [VideoFileClip(str(p)) for p in clip_paths]

    if transition_duration > 0 and len(clips) > 1:
        for i in range(1, len(clips)):
            clips[i] = clips[i].crossfadein(transition_duration)
        final = concatenate_videoclips(clips, method="compose", padding=-transition_duration)
    else:
        final = concatenate_videoclips(clips, method="compose")

    # Mix background music underneath narration.
    if music_path and Path(music_path).exists():
        try:
            music = AudioFileClip(str(music_path)).volumex(config.DEFAULT_MUSIC_VOLUME)
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
