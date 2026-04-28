"""
Text-to-speech tool. Tries ElevenLabs first, falls back to Coqui TTS.

The cache key includes the text + emotion + provider, and the cached file
keeps its native extension (.mp3 for ElevenLabs, .wav for Coqui) so a cache
hit returns a path that actually exists. The downstream merge_audio_video
in tools/video_utils.py accepts either format.
"""
import logging
import hashlib
import wave
import contextlib
from pathlib import Path
from typing import Optional
from config import config

logger = logging.getLogger(__name__)


class TTSEngine:
    """Unified TTS wrapper."""

    def __init__(self):
        self._coqui = None
        self._elevenlabs = None

    def _load_coqui(self):
        if self._coqui is not None:
            return
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError("Run: pip install TTS")
        logger.info(f"Loading Coqui TTS: {config.COQUI_MODEL}")
        self._coqui = TTS(config.COQUI_MODEL, progress_bar=False)

    def _load_elevenlabs(self):
        if self._elevenlabs is not None:
            return
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError:
            raise ImportError("Run: pip install elevenlabs")
        self._elevenlabs = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)

    def synthesize(
        self,
        text: str,
        output_path: Path,
        emotion: str = "neutral",
        cache: bool = True,
    ) -> Path:
        """Generate speech audio and return the file path."""
        # Pick the provider up front so the cache key matches the file extension.
        provider = config.TTS_PROVIDER
        use_eleven = provider == "elevenlabs" and config.ELEVENLABS_API_KEY
        ext = "mp3" if use_eleven else "wav"

        if cache:
            key = hashlib.sha256(f"{provider}|{text}|{emotion}".encode()).hexdigest()[:16]
            cache_path = config.CACHE_DIR / f"tts_{key}.{ext}"
            if cache_path.exists():
                logger.info(f"TTS cache hit: {cache_path.name}")
                return cache_path
            output_path = cache_path

        if use_eleven:
            try:
                return self._synth_elevenlabs(text, output_path)
            except Exception as e:
                logger.warning(f"ElevenLabs failed ({e}), falling back to Coqui.")
                # Coqui produces .wav — adjust path so the file exists at the
                # extension we return.
                return self._synth_coqui(text, output_path.with_suffix(".wav"))
        return self._synth_coqui(text, output_path)

    def _synth_elevenlabs(self, text: str, output_path: Path) -> Path:
        self._load_elevenlabs()
        audio_stream = self._elevenlabs.text_to_speech.convert(
            voice_id=config.ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                if chunk:
                    f.write(chunk)
        logger.info(f"ElevenLabs audio saved: {output_path.name}")
        return output_path

    def _synth_coqui(self, text: str, output_path: Path) -> Path:
        self._load_coqui()
        wav_path = output_path.with_suffix(".wav")
        self._coqui.tts_to_file(text=text, file_path=str(wav_path))
        logger.info(f"Coqui audio saved: {wav_path.name}")
        return wav_path


_singleton: Optional[TTSEngine] = None

def get_tts() -> TTSEngine:
    global _singleton
    if _singleton is None:
        _singleton = TTSEngine()
    return _singleton


def get_audio_duration(path: Path) -> float:
    """Return audio length in seconds. Falls back to MoviePy if not a WAV."""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            with contextlib.closing(wave.open(str(p), "rb")) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        except Exception:
            pass
    # Fallback: pull duration via MoviePy (handles mp3/m4a/etc).
    from moviepy.editor import AudioFileClip
    clip = AudioFileClip(str(p))
    try:
        return float(clip.duration)
    finally:
        clip.close()
