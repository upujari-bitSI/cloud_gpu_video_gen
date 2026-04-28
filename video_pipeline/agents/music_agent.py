"""
Music Agent: picks a background music track based on the overall mood.
Reads .mp3 files from assets/music/ and matches by filename keywords.
For production: integrate with an AI music API like Suno/Udio/Mubert.
"""
import random
from pathlib import Path
from collections import Counter
from agents.base import BaseAgent
from config import config


# mood-keyword -> filename keyword to look for
MOOD_KEYWORDS = {
    "tense": ["tense", "dark", "thriller"],
    "hopeful": ["hopeful", "uplifting", "warm"],
    "melancholic": ["sad", "melancholic", "emotional"],
    "exciting": ["epic", "action", "energetic"],
    "mysterious": ["mystery", "ambient", "ethereal"],
    "happy": ["happy", "uplifting", "cheerful"],
    "neutral": ["ambient", "cinematic"],
}


class MusicAgent(BaseAgent):
    name = "MusicAgent"

    async def run(self, state):
        music_dir = config.MUSIC_LIBRARY_DIR
        if not music_dir.exists() or not any(music_dir.glob("*.mp3")):
            self.logger.warning(
                f"No music files found in {music_dir}. "
                "Drop .mp3 files there or final video will have no music."
            )
            return state

        # Determine dominant mood from scenes
        moods = [s.mood.lower() for s in state.scenes if s.mood]
        dominant_mood = Counter(moods).most_common(1)[0][0] if moods else "neutral"
        self.logger.info(f"Dominant mood: {dominant_mood}")

        keywords = MOOD_KEYWORDS.get(dominant_mood, ["cinematic"])
        candidates = []
        for kw in keywords:
            candidates.extend(music_dir.glob(f"*{kw}*.mp3"))

        if not candidates:
            candidates = list(music_dir.glob("*.mp3"))

        chosen = random.choice(candidates)
        state.music_path = str(chosen)
        self.logger.info(f"Music: {chosen.name}")
        return state
