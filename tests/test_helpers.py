# tests/test_helpers.py
from collections.abc import Sequence
from pathlib import Path
import re

from domain.errors import Result, audio_generation_error, tts_engine_error
from domain.interfaces import (
    IAudioEncoder,
    ITTSEngine,
)
from domain.models import (
    SynthesizedSegment,
)


def fake_segments(text: str, sample_rate: int = 100) -> list[SynthesizedSegment]:
    """Fake segments for a piece of text, one per sentence.

    Durations scale with sentence length, so tests can tell a short sentence from
    a long one - which is exactly what the old even-split timing could not do.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()] or [text]
    return [
        SynthesizedSegment(
            text=sentence,
            pcm=b"\x01\x00" * max(1, len(sentence)),
            sample_rate=sample_rate,
            sample_width=2,
            channels=1,
        )
        for sentence in sentences
    ]


class FakeTTSEngine(ITTSEngine):
    """Fake TTS engine for testing purposes."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.generated_texts: list[str] = []

    def synthesize(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Produce one fake segment per sentence, sized by sentence length."""
        self.generated_texts.append(text)
        if self.should_fail:
            return Result.failure(tts_engine_error("TTS generation failed"))
        return Result.success(fake_segments(text))

    async def synthesize_async(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Async version for interface compliance."""
        return self.synthesize(text)

    def check_ready(self) -> Result[None]:
        """Ready unless built to fail."""
        if self.should_fail:
            return Result.failure(tts_engine_error("TTS engine not ready"))
        return Result.success(None)


class FakeAudioEncoder(IAudioEncoder):
    """Writes the WAV bytes unchanged to the MP3 path, so tests can decode the output."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.encoded: dict[str, bytes] = {}

    def encode_mp3(self, wav: bytes, dest_path: str) -> Result[str]:
        """Record and write the WAV, or fail if built to."""
        if self.should_fail:
            return Result.failure(audio_generation_error("encoding failed"))
        Path(dest_path).write_bytes(wav)
        self.encoded[dest_path] = wav
        return Result.success(dest_path)
