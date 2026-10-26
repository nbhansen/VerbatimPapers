"""AudioEngine: one pipeline from text chunks to an MP3 and its read-along timeline."""

import asyncio
from collections.abc import Sequence
import io
from pathlib import Path
import wave

import pytest

from domain.audio.audio_assembler import AudioAssembler
from domain.audio.audio_engine import AudioEngine
from domain.errors import Result
from domain.models import SynthesizedSegment
from infrastructure.file.file_manager import FileManager
from tests.test_helpers import FakeAudioEncoder, FakeTTSEngine, fake_segments


class OutOfOrderTTSEngine(FakeTTSEngine):
    """Finishes later chunks first, so completion order is the reverse of input order."""

    def __init__(self, chunk_count: int):
        super().__init__()
        self.chunk_count = chunk_count
        self.in_flight = 0
        self.max_in_flight = 0

    async def synthesize_async(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Delay inversely to the chunk's position, read from its 'Chunk N.' text."""
        index = int(text.split()[1].rstrip("."))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01 * (self.chunk_count - index))
        self.in_flight -= 1
        return self.synthesize(text)


class FailingChunkTTSEngine(FakeTTSEngine):
    """Fails every chunk containing the word FAIL."""

    async def synthesize_async(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Fail on marked chunks, otherwise behave like the fake engine."""
        if "FAIL" in text:
            return FakeTTSEngine(should_fail=True).synthesize(text)
        return self.synthesize(text)


@pytest.fixture
def file_manager(tmp_path: Path) -> FileManager:
    """File manager writing into a temp directory."""
    return FileManager(upload_folder=str(tmp_path / "uploads"), output_folder=str(tmp_path / "audio"))


def _engine(
    file_manager: FileManager,
    tts: FakeTTSEngine | None = None,
    encoder: FakeAudioEncoder | None = None,
    max_concurrent: int = 4,
    chunk_size: int = 30,
) -> AudioEngine:
    return AudioEngine(
        tts_engine=tts or FakeTTSEngine(),
        file_manager=file_manager,
        encoder=encoder or FakeAudioEncoder(),
        assembler=AudioAssembler(sentence_gap=0.1),
        max_concurrent=max_concurrent,
        audio_target_chunk_size=chunk_size,
    )


def _wav_duration(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


class TestGenerate:
    """generate() returns the audio and its timing together."""

    def test_timeline_matches_audio_duration(self, file_manager: FileManager) -> None:
        """The timeline's total equals the duration of the audio actually written."""
        encoder = FakeAudioEncoder()
        chunks = ["A short one. Then a much longer sentence follows it.", "Another chunk here. And the end."]

        result = _engine(file_manager, encoder=encoder).generate(chunks, "doc")

        assert result.is_success and result.value is not None and result.value.timing_data is not None
        timing = result.value.timing_data
        (wav,) = encoder.encoded.values()
        assert timing.total_duration == pytest.approx(_wav_duration(wav))
        last = timing.text_segments[-1]
        assert last.start_time + last.duration + 0.1 == pytest.approx(timing.total_duration)

    def test_output_order_survives_concurrency(self, file_manager: FileManager) -> None:
        """Chunks finishing out of order still play and highlight in document order."""
        texts = [f"Chunk {i}." for i in range(6)]
        tts = OutOfOrderTTSEngine(chunk_count=len(texts))
        encoder = FakeAudioEncoder()

        result = _engine(file_manager, tts=tts, encoder=encoder, max_concurrent=3, chunk_size=10).generate(texts, "doc")

        assert result.is_success and result.value is not None and result.value.timing_data is not None
        assert tts.max_in_flight == 3
        segments = result.value.timing_data.text_segments
        assert [s.text for s in segments] == texts
        assert [s.chunk_index for s in segments] == list(range(6))
        expected_pcm = b"".join(fake_segments(t)[0].pcm + bytes(10 * 2) for t in texts)
        (wav,) = encoder.encoded.values()
        with wave.open(io.BytesIO(wav), "rb") as wav_file:
            assert wav_file.readframes(wav_file.getnframes()) == expected_pcm

    def test_writes_one_mp3_named_after_output(self, file_manager: FileManager) -> None:
        """The MP3 lands in the output folder, and the timing names the same file."""
        result = _engine(file_manager).generate(["Hello there."], "paper_ab12cd34")

        assert result.is_success and result.value is not None and result.value.timing_data is not None
        assert result.value.audio_files == ["paper_ab12cd34.mp3"]
        assert result.value.combined_mp3 == "paper_ab12cd34.mp3"
        assert result.value.timing_data.audio_files == ["paper_ab12cd34.mp3"]
        assert (Path(file_manager.get_output_dir()) / "paper_ab12cd34.mp3").exists()

    def test_failed_chunk_fails_the_conversion(self, file_manager: FileManager) -> None:
        """One unsynthesizable chunk fails the whole conversion rather than leaving a silent gap."""
        encoder = FakeAudioEncoder()

        result = _engine(file_manager, tts=FailingChunkTTSEngine(), encoder=encoder, chunk_size=10).generate(
            ["First one.", "FAIL here.", "Last one."], "doc"
        )

        assert result.is_failure and result.error is not None
        assert result.error.details == "Part 2 of 3 could not be synthesized: TTS generation failed"
        assert encoder.encoded == {}, "No audio may be published with a passage missing"

    def test_all_chunks_failing_is_an_error(self, file_manager: FileManager) -> None:
        """No audio at all is a failure, not an empty success."""
        result = _engine(file_manager, tts=FakeTTSEngine(should_fail=True)).generate(["Hello."], "doc")

        assert result.is_failure and result.error is not None
        assert "Part 1 of 1 could not be synthesized" in (result.error.details or "")

    def test_encoder_failure_is_an_error(self, file_manager: FileManager) -> None:
        """An encoding failure is reported rather than returning a missing file."""
        result = _engine(file_manager, encoder=FakeAudioEncoder(should_fail=True)).generate(["Hello."], "doc")

        assert result.is_failure

    def test_blank_text_is_an_error(self, file_manager: FileManager) -> None:
        """Nothing to say is reported, not synthesized."""
        result = _engine(file_manager).generate(["   ", ""], "doc")

        assert result.is_failure
