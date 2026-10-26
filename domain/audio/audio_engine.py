# domain/audio/audio_engine.py - Audio Processing Engine with Result[T] pattern
"""Audio engine: text chunks in, one MP3 and its read-along timeline out.

No exceptions thrown - all errors returned as Result[T].

Timing Algorithm
================

Timestamps for read-along come from the segments the TTS engine returns, not
from estimation.

1. Synthesize each text chunk, which yields one segment per sentence carrying
   both the sentence text and its audio.
2. Lay the segments out on a timeline using their measured durations, plus the
   inter-sentence gap the assembler will write between them.
3. Assemble all segments into one audio file.

Because the same AudioAssembler decides both the gap in the timeline and the gap
in the audio, the two cannot drift apart.

Engines with no internal structure (Gemini) return a single segment per chunk, so
timing is chunk-granular rather than sentence-granular. That is a real limitation
of the engine, and it is visible in the data rather than hidden behind an
estimate that looks precise.
"""

import asyncio
from collections.abc import Sequence
import logging
from pathlib import Path

from ..errors import ErrorCode, Result, audio_generation_error
from ..interfaces import IAudioEncoder, IAudioEngine, IFileManager, ITTSEngine
from ..models import SynthesizedSegment, TextSegment, TimedAudioResult, TimingMetadata
from ..text.segmentation import chunk_text
from .audio_assembler import AudioAssembler

logger = logging.getLogger(__name__)


class AudioEngine(IAudioEngine):
    """Synthesizes chunks concurrently and assembles them in document order.

    Pacing for rate-limited engines is the engine adapter's job; this class only
    bounds how many chunks are in flight at once.
    """

    def __init__(
        self,
        tts_engine: ITTSEngine,
        file_manager: IFileManager,
        encoder: IAudioEncoder,
        assembler: AudioAssembler | None = None,
        max_concurrent: int = 4,
        audio_target_chunk_size: int = 2000,
    ):
        self.tts_engine = tts_engine
        self.file_manager = file_manager
        self.encoder = encoder
        self.assembler = assembler or AudioAssembler()
        self.max_concurrent = max(1, max_concurrent)
        self.audio_target_chunk_size = audio_target_chunk_size

    def generate(self, text_chunks: list[str], output_filename: str) -> Result[TimedAudioResult]:
        """Synthesize the text into <output_filename>.mp3 with a measured timeline.

        This is the only place text is chunked for TTS: the input is re-packed into
        sentence-aligned chunks of at most audio_target_chunk_size characters.
        """
        chunks = chunk_text(" ".join(text_chunks), self.audio_target_chunk_size)
        if not chunks:
            return Result.failure(audio_generation_error("No text to synthesize"))

        logger.info("Synthesizing %d chunks (max size %d chars)", len(chunks), self.audio_target_chunk_size)

        try:
            results = asyncio.run(self._synthesize_all(chunks))
        except Exception as e:
            logger.exception("Synthesis failed")
            return Result.from_exception(e, ErrorCode.AUDIO_GENERATION_FAILED, retryable=True)

        all_segments: list[SynthesizedSegment] = []
        timed_segments: list[TextSegment] = []
        cursor = 0.0
        # gather() returns results in input order, so the timeline follows the document
        for chunk_index, result in enumerate(results):
            if result.is_failure or not result.value:
                logger.warning("Chunk %d/%d failed: %s", chunk_index + 1, len(chunks), result.error)
                continue
            segments = list(result.value)
            chunk_timed, cursor = self.assembler.to_timed_segments(segments, chunk_index=chunk_index, start_time=cursor)
            all_segments.extend(segments)
            timed_segments.extend(chunk_timed)

        if not all_segments:
            return Result.failure(audio_generation_error("No audio generated for any chunk"))

        assembled = self.assembler.to_wav(all_segments)
        if assembled.is_failure or not assembled.value:
            return Result.failure(assembled.error or audio_generation_error("Failed to assemble audio"))

        mp3_filename = f"{output_filename}.mp3"
        mp3_path = Path(self.file_manager.get_output_dir()) / mp3_filename
        encoded = self.encoder.encode_mp3(assembled.value, str(mp3_path))
        if encoded.is_failure:
            return Result.failure(encoded.error or audio_generation_error("MP3 encoding failed"))

        logger.info("Generated %s: %d segments over %.2fs", mp3_filename, len(timed_segments), cursor)
        return Result.success(
            TimedAudioResult(
                audio_files=[mp3_filename],
                combined_mp3=mp3_filename,
                timing_data=TimingMetadata(
                    total_duration=cursor, text_segments=timed_segments, audio_files=[mp3_filename]
                ),
            )
        )

    async def _synthesize_all(self, chunks: list[str]) -> list[Result[Sequence[SynthesizedSegment]]]:
        """Synthesize every chunk, at most max_concurrent at a time, keeping input order."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def synthesize(chunk: str) -> Result[Sequence[SynthesizedSegment]]:
            async with semaphore:
                try:
                    return await self.tts_engine.synthesize_async(chunk)
                except Exception as e:
                    logger.exception("TTS engine raised instead of returning a Result")
                    return Result.from_exception(e, ErrorCode.TTS_ENGINE_ERROR, retryable=True)

        return list(await asyncio.gather(*(synthesize(chunk) for chunk in chunks)))
