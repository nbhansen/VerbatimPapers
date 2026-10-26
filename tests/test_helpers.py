# tests/test_helpers.py
from collections.abc import Sequence
from pathlib import Path
import re
import tempfile

from domain.audio.timing_engine import ITimingEngine
from domain.errors import Result, llm_provider_error, tts_engine_error
from domain.interfaces import (
    ILLMProvider,
    ITTSEngine,
)
from domain.models import (
    PageRange,
    ProcessingRequest,
    SynthesizedSegment,
    TimedAudioResult,
    TimingMetadata,
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

    def supports_ssml(self) -> bool:
        """Return whether this engine supports SSML."""
        return False

    def check_ready(self) -> Result[None]:
        """Ready unless built to fail."""
        if self.should_fail:
            return Result.failure(tts_engine_error("TTS engine not ready"))
        return Result.success(None)


class FakeLLMProvider(ILLMProvider):
    """Fake LLM provider for testing purposes."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.prompts: list[str] = []

    def generate_content(self, prompt: str) -> Result[str]:
        """Generate fake content for testing."""
        self.prompts.append(prompt)
        if self.should_fail:
            return Result.failure(llm_provider_error("LLM generation failed"))
        return Result.success(f"Cleaned: {prompt[:50]}... with pauses")


def create_test_request(pdf_path="test.pdf", output_name="test_output", page_range=None):
    return ProcessingRequest(pdf_path=pdf_path, output_name=output_name, page_range=page_range or PageRange())


class FakeFileManager:
    """Fake file manager for testing."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir or tempfile.mkdtemp()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.saved_files: list[str] = []
        self.temp_files: list[str] = []

    def save_output_file(self, content: bytes, filename: str) -> str:
        """Save content to output file for testing."""
        filepath = Path(self.output_dir) / filename
        with filepath.open("wb") as f:
            f.write(content)
        self.saved_files.append(str(filepath))
        return str(filepath)

    def save_temp_file(self, content: bytes, suffix: str = ".tmp") -> str:
        """Save content to temporary file for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(content)
            self.temp_files.append(f.name)
            return f.name

    def get_output_dir(self) -> str:
        """Return the output directory path."""
        return self.output_dir

    def delete_file(self, filepath: str) -> None:
        """Delete a file if it exists."""
        if Path(filepath).exists():
            Path(filepath).unlink()
            if filepath in self.saved_files:
                self.saved_files.remove(filepath)
            if filepath in self.temp_files:
                self.temp_files.remove(filepath)

    def cleanup(self):
        """Clean up all created files."""
        for filepath in self.saved_files + self.temp_files:
            if Path(filepath).exists():
                Path(filepath).unlink()
        self.saved_files.clear()
        self.temp_files.clear()


class FakeTimingEngine(ITimingEngine):
    """Fake timing engine for testing."""

    def generate_with_timing(self, text_chunks: list[str], output_filename: str) -> Result[TimedAudioResult]:
        """Generate fake timed audio result for testing."""
        from domain.models import TextSegment

        # Create fake segments
        segments = []
        current_time = 0.0

        for i, chunk in enumerate(text_chunks):
            duration = len(chunk.split()) * 0.5  # Half second per word
            segment = TextSegment(
                text=chunk,
                start_time=current_time,
                duration=duration,
                segment_type="sentence",
                chunk_index=i,
                sentence_index=i,
            )
            segments.append(segment)
            current_time += duration

        timing_metadata = TimingMetadata(
            total_duration=current_time, text_segments=segments, audio_files=[f"{output_filename}.wav"]
        )

        return Result.success(
            TimedAudioResult(
                audio_files=[f"{output_filename}.wav"],
                combined_mp3=f"{output_filename}.mp3",
                timing_data=timing_metadata,
            )
        )


# Alias for backward compatibility with tests
FakeTimingStrategy = FakeTimingEngine
