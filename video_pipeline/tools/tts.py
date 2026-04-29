"""
Text-to-speech tool. Tries providers in priority order: kokoro -> elevenlabs
-> coqui. Each is auto-skipped if its package isn't installed or its loader
fails — pipeline never fails just because one TTS engine is missing.

Provider selection is via config.TTS_PROVIDER. "kokoro" is the default
(82M-param open-weight model, runs comfortably on CPU or any GPU, much more
natural than the original Tacotron2-DDC).

Cache key includes provider so swapping engines mid-project doesn't reuse
the previous engine's audio.
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
        self._kokoro = None  # KPipeline instance

    def _load_kokoro(self):
        if self._kokoro is not None:
            return
        from kokoro import KPipeline  # raises ImportError if not installed
        logger.info(f"Loading Kokoro TTS (lang={config.KOKORO_LANG_CODE}, voice={config.KOKORO_VOICE})")
        self._kokoro = KPipeline(lang_code=config.KOKORO_LANG_CODE)

    def _load_coqui(self):
        if self._coqui is not None:
            return
        from TTS.api import TTS  # raises ImportError if not installed
        logger.info(f"Loading Coqui TTS: {config.COQUI_MODEL}")
        self._coqui = TTS(config.COQUI_MODEL, progress_bar=False)

    def _load_elevenlabs(self):
        if self._elevenlabs is not None:
            return
        from elevenlabs.client import ElevenLabs
        self._elevenlabs = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)

    def synthesize(
        self,
        text: str,
        output_path: Path,
        emotion: str = "neutral",
        cache: bool = True,
    ) -> Path:
        """Generate speech audio and return the file path."""
        provider = (config.TTS_PROVIDER or "kokoro").lower()

        # Cache key encodes provider + voice + text so changing provider
        # forces regeneration rather than returning a stale clip.
        voice_key = config.KOKORO_VOICE if provider == "kokoro" else provider
        key = hashlib.sha256(
            f"{provider}|{voice_key}|{text}|{emotion}".encode()
        ).hexdigest()[:16]
        cache_path = config.CACHE_DIR / f"tts_{key}.wav"
        if cache and cache_path.exists():
            logger.info(f"TTS cache hit: {cache_path.name}")
            return cache_path
        if cache:
            output_path = cache_path

        # Try the requested provider first, then fall through to alternatives.
        order = {
            "kokoro": ["kokoro", "elevenlabs", "coqui"],
            "elevenlabs": ["elevenlabs", "kokoro", "coqui"],
            "coqui": ["coqui", "kokoro"],
        }.get(provider, ["kokoro", "coqui"])

        last_err: Optional[Exception] = None
        for prov in order:
            if prov == "elevenlabs" and not config.ELEVENLABS_API_KEY:
                continue
            try:
                if prov == "kokoro":
                    return self._synth_kokoro(text, output_path)
                if prov == "elevenlabs":
                    return self._synth_elevenlabs(text, output_path.with_suffix(".mp3"))
                if prov == "coqui":
                    return self._synth_coqui(text, output_path.with_suffix(".wav"))
            except Exception as e:
                logger.warning(f"TTS provider '{prov}' failed: {e}")
                last_err = e
                continue
        raise RuntimeError(f"All TTS providers failed; last error: {last_err}")

    def _synth_kokoro(self, text: str, output_path: Path) -> Path:
        """Kokoro-82M synthesis. Returns a single concatenated WAV.

        Kokoro yields per-sentence chunks; we concatenate them into one
        clip so the rest of the pipeline (which assumes one file per scene)
        works unchanged.
        """
        self._load_kokoro()
        import numpy as np
        import soundfile as sf

        chunks = []
        for _, _, audio in self._kokoro(
            text,
            voice=config.KOKORO_VOICE,
            speed=config.KOKORO_SPEED,
        ):
            # Audio comes back as a torch tensor in current Kokoro releases.
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        wav = np.concatenate(chunks)
        wav_path = output_path.with_suffix(".wav")
        sf.write(str(wav_path), wav, 24000)  # Kokoro outputs 24 kHz
        logger.info(f"Kokoro audio saved: {wav_path.name}")
        return wav_path

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
    """Return audio length in seconds.

    Fast path: read header for WAV files via the stdlib `wave` module.
    Fallback: probe with ffmpeg (bundled by imageio_ffmpeg, no system install
    needed) and parse the Duration line. Avoids pulling in moviepy here.
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            with contextlib.closing(wave.open(str(p), "rb")) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        except Exception:
            pass

    import subprocess
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    # ffmpeg writes media info to stderr even on the "null" probe.
    result = subprocess.run(
        [ffmpeg_bin, "-i", str(p), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for line in result.stderr.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            # "Duration: 00:00:11.58, start: ..."
            ts = line.split(",", 1)[0].split("Duration:", 1)[1].strip()
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"Could not determine duration of {p}")
