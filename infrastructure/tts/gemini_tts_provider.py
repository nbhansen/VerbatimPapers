# infrastructure/tts/gemini_tts_provider.py
"""Gemini TTS Provider implementation using Google Gen AI SDK."""

import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import io
import logging
import threading
import time
import wave

from google import genai
from google.genai import types

from domain.errors import Result, tts_engine_error
from domain.interfaces import ITTSEngine
from domain.models import SynthesizedSegment
from infrastructure.gemini_http import default_http_options

logger = logging.getLogger(__name__)

# Gemini takes delivery direction as natural language rather than SSML, so this is the
# only prosody control the engine exposes. Tuned for long-form academic reading.
DEFAULT_STYLE_PROMPT = (
    "Read the following text aloud in a measured, clear academic tone. "
    "Pace it for comprehension rather than speed, and pause between sections."
)


class GeminiTTSProvider(ITTSEngine):
    """Gemini TTS Provider using Google Gen AI SDK."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        voice_name: str = "Kore",
        style_prompt: str | None = None,
        min_request_interval: float = 2.0,
        max_concurrent_requests: int = 4,
        http_options: types.HttpOptions | None = None,
    ):
        """Initialize Gemini TTS Provider.

        Args:
            model_name: Gemini speech model (a *-tts variant, e.g. gemini-2.5-flash-preview-tts)
            api_key: Google AI API Key
            voice_name: Voice to use (e.g. "Kore", "Puck", "Charon", "Fenrir", "Aoede")
            style_prompt: Delivery directive prefixed to the text. None uses
                DEFAULT_STYLE_PROMPT; pass an empty string to send the text bare.
            min_request_interval: Minimum seconds between the starts of API requests
            max_concurrent_requests: Maximum number of API requests in flight
            http_options: Client HTTP settings; defaults to retrying transient errors

        Both limits are enforced across every caller of this instance, including
        conversions running concurrently on separate threads and event loops.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.voice_name = voice_name
        self.style_prompt = DEFAULT_STYLE_PROMPT if style_prompt is None else style_prompt
        self.min_request_interval = min_request_interval
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        self._request_slots = threading.BoundedSemaphore(self.max_concurrent_requests)
        self._pace_lock = threading.Lock()
        self._next_request_at = 0.0
        self._initialization_error: str | None = None  # Deferred error handling

        self.client: genai.Client | None = None
        try:
            self.client = genai.Client(api_key=self.api_key, http_options=http_options or default_http_options())
            logger.info("Gemini TTS initialized with model=%s, voice=%s", model_name, voice_name)
        except Exception as e:
            self._initialization_error = f"Failed to initialize Gemini Client: {e}"
            logger.error(self._initialization_error)

    def _claim_start_delay(self) -> float:
        """Reserve the next request start time and return how long to wait for it."""
        with self._pace_lock:
            now = time.monotonic()
            start = max(now, self._next_request_at)
            self._next_request_at = start + self.min_request_interval
            return start - now

    @contextmanager
    def _request_slot(self) -> Iterator[None]:
        """Hold one in-flight request slot, paced by min_request_interval (blocking)."""
        with self._request_slots:
            time.sleep(self._claim_start_delay())
            yield

    def _build_prompt(self, text: str) -> str:
        """Prefix the delivery directive to the text, if one is configured."""
        if not self.style_prompt:
            return text
        return f"{self.style_prompt}\n\n{text}"

    def synthesize(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Synthesize text into a single segment.

        Gemini returns one undifferentiated stretch of audio with no sentence
        boundaries, so there is exactly one segment covering the whole text.
        """
        # Check for deferred initialization errors
        if self._initialization_error:
            logger.error("Gemini TTS initialization failed: %s", self._initialization_error)
            return Result.failure(tts_engine_error(self._initialization_error))

        if not self.client:
            return Result.failure(tts_engine_error("Gemini Client not initialized"))

        if not text.strip():
            return Result.failure(tts_engine_error("Empty text provided"))

        logger.debug("Generating audio for %d chars of text", len(text))
        try:
            # Configure for audio generation
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice_name)
                    )
                ),
            )

            prompt = self._build_prompt(text)

            with self._request_slot():
                response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=config)
            logger.debug("Gemini API call successful, extracting audio from response")

            return self._segments_from_response(text, response)

        except Exception as e:
            logger.exception("Gemini TTS generation failed")
            return Result.failure(tts_engine_error(f"Gemini TTS failed: {e}"))

    async def synthesize_async(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Run the sync client in a worker thread.

        Each conversion runs on its own short-lived event loop. genai's async client
        keeps one connection pool for the process, and pooled connections are tied to
        the loop that opened them, so reusing it on a later loop fails with "Event
        loop is closed". The sync client has no such tie.
        """
        return await asyncio.to_thread(self.synthesize, text)

    def _segments_from_response(
        self, text: str, response: types.GenerateContentResponse
    ) -> Result[Sequence[SynthesizedSegment]]:
        """Build a single segment from a Gemini response."""
        num_candidates = len(response.candidates) if response.candidates else 0
        logger.debug("Extracting audio from response (candidates=%d)", num_candidates)
        if not response.candidates:
            return Result.failure(tts_engine_error("No candidates returned from Gemini"))

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return Result.failure(tts_engine_error("Empty content in Gemini response"))

        for part in candidate.content.parts:
            # Check for inline_data with audio mime type
            if (
                part.inline_data
                and part.inline_data.mime_type
                and part.inline_data.mime_type.startswith("audio")
                and part.inline_data.data
            ):
                audio_data = part.inline_data.data
                logger.debug(
                    "Found audio part: mime_type=%s, size=%d bytes",
                    part.inline_data.mime_type,
                    len(audio_data),
                )
                if "pcm" in part.inline_data.mime_type:
                    sample_rate = self._parse_sample_rate(part.inline_data.mime_type)
                    logger.debug("PCM audio detected (sample_rate=%d)", sample_rate)
                    return Result.success(
                        [
                            SynthesizedSegment(
                                text=text,
                                pcm=audio_data,
                                sample_rate=sample_rate,
                                sample_width=2,  # Gemini PCM is 16-bit
                                channels=1,
                            )
                        ]
                    )

                # Anything else arrives as an encoded container we cannot treat as PCM
                return self._segment_from_container(text, audio_data)

        return Result.failure(tts_engine_error("No audio data found in Gemini response"))

    @staticmethod
    def _parse_sample_rate(mime_type: str, default: int = 24000) -> int:
        """Read the sample rate out of a PCM mime type, falling back to the default."""
        if "rate=" not in mime_type:
            return default
        try:
            return int(mime_type.split("rate=")[1].split(";")[0])
        except (IndexError, ValueError):
            return default

    @staticmethod
    def _segment_from_container(text: str, audio_data: bytes) -> Result[Sequence[SynthesizedSegment]]:
        """Unwrap a WAV container into a segment; other formats are unsupported."""
        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
                return Result.success(
                    [
                        SynthesizedSegment(
                            text=text,
                            pcm=wav_file.readframes(wav_file.getnframes()),
                            sample_rate=wav_file.getframerate(),
                            sample_width=wav_file.getsampwidth(),
                            channels=wav_file.getnchannels(),
                        )
                    ]
                )
        except (wave.Error, EOFError) as e:
            return Result.failure(tts_engine_error(f"Gemini returned audio in an unsupported format: {e}"))

    def check_ready(self) -> Result[None]:
        """Ready once the client is built; the service itself is only reached on first request."""
        if self._initialization_error:
            return Result.failure(tts_engine_error(self._initialization_error))
        if not self.client:
            return Result.failure(tts_engine_error("Gemini Client not initialized"))
        return Result.success(None)
