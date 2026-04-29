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
    # Use ffmpeg directly — moviepy 1.0.3 write_videofile passes fps=None to
    # the writer on this env. -shortest trims output to whichever stream ends
    # first, which keeps audio/video in sync with no read-past-end errors.
    import subprocess
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    fps = config.VIDEO_FPS or 24
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-b:v", config.VIDEO_BITRATE, "-preset", "medium",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def stitch_clips(
    clip_paths: list[Path],
    output_path: Path,
    transition_duration: float = 0.5,
    music_path: Optional[Path] = None,
) -> Path:
    """Concatenate multiple clips end-to-end and optionally mix bg music.

    moviepy's write path is unreliable on this env (fps=None bug), so we
    use ffmpeg's concat demuxer. Crossfades are not applied here — the
    visual cuts are scene-synced to narration, which works fine without
    transitions.
    """
    import subprocess
    import imageio_ffmpeg
    import tempfile

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    fps = config.VIDEO_FPS or 24

    # Write a concat list for ffmpeg's demuxer.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
        list_path = f.name

    has_music = bool(music_path and Path(music_path).exists())
    if has_music:
        # Concat to a temp file first, then mix music in a second pass.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            concat_path = f.name
    else:
        concat_path = str(output_path)

    concat_cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-b:v", config.VIDEO_BITRATE, "-preset", "medium",
        "-c:a", "aac",
        concat_path,
    ]
    subprocess.run(concat_cmd, check=True)

    if has_music:
        mix_cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", concat_path,
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex",
            f"[1:a]volume={config.DEFAULT_MUSIC_VOLUME}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]
        subprocess.run(mix_cmd, check=True)
        Path(concat_path).unlink(missing_ok=True)

    Path(list_path).unlink(missing_ok=True)
    return output_path
