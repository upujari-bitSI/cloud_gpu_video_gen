"""
Text-to-speech tool. Tries ElevenLabs first, falls back to Coqui TTS.
"""
import logging
import hashlib
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
        if cache:
            key = hashlib.sha256(f"{text}{emotion}".encode()).hexdigest()[:16]
            cache_path = config.CACHE_DIR / f"tts_{key}.mp3"
            if cache_path.exists():
                logger.info(f"TTS cache hit: {cache_path.name}")
                return cache_path
            output_path = cache_path

        provider = config.TTS_PROVIDER
        if provider == "elevenlabs" and config.ELEVENLABS_API_KEY:
            try:
                return self._synth_elevenlabs(text, output_path)
            except Exception as e:
                logger.warning(f"ElevenLabs failed ({e}), falling back to Coqui.")
                return self._synth_coqui(text, output_path)
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
        # Coqui outputs WAV; we save as wav and rename or convert
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
