# infrastructure/tts/gemini_tts_provider.py
"""Gemini TTS Provider implementation using Google Gen AI SDK."""

from collections.abc import Sequence
import io
import logging
import wave

from google import genai
from google.genai import types

from domain.errors import Result, tts_engine_error
from domain.interfaces import ITTSEngine
from domain.models import SynthesizedSegment

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
        requests_per_minute: int = 30,
    ):
        """Initialize Gemini TTS Provider.

        Args:
            model_name: Gemini model (e.g. gemini-2.0-flash-exp)
            api_key: Google AI API Key
            voice_name: Voice to use (e.g. "Kore", "Puck", "Charon", "Fenrir", "Aoede")
            style_prompt: Delivery directive prefixed to the text. None uses
                DEFAULT_STYLE_PROMPT; pass an empty string to send the text bare.
            min_request_interval: Minimum seconds between API requests
            max_concurrent_requests: Maximum number of concurrent API requests
            requests_per_minute: Rate limit for API requests per minute
        """
        self.model_name = model_name
        self.api_key = api_key
        self.voice_name = voice_name
        self.style_prompt = DEFAULT_STYLE_PROMPT if style_prompt is None else style_prompt
        self.min_request_interval = min_request_interval
        self.max_concurrent_requests = max_concurrent_requests
        self.requests_per_minute = requests_per_minute
        self.output_format = "mp3"  # Gemini audio is typically MP3
        self._initialization_error: str | None = None  # Deferred error handling

        self.client: genai.Client | None = None
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Gemini TTS initialized with model=%s, voice=%s", model_name, voice_name)
        except Exception as e:
            self._initialization_error = f"Failed to initialize Gemini Client: {e}"
            logger.error(self._initialization_error)

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

            response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=config)
            logger.debug("Gemini API call successful, extracting audio from response")

            return self._segments_from_response(text, response)

        except Exception as e:
            logger.exception("Gemini TTS generation failed")
            return Result.failure(tts_engine_error(f"Gemini TTS failed: {e}"))

    async def synthesize_async(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Synthesize asynchronously using Gemini's async client."""
        # Check for deferred initialization errors
        if self._initialization_error:
            logger.error("Gemini TTS initialization failed: %s", self._initialization_error)
            return Result.failure(tts_engine_error(self._initialization_error))

        if not self.client:
            return Result.failure(tts_engine_error("Gemini Client not initialized"))

        try:
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice_name)
                    )
                ),
            )

            prompt = self._build_prompt(text)

            # Use async client
            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=prompt, config=config
            )

            return self._segments_from_response(text, response)

        except Exception as e:
            logger.exception("Gemini Async TTS generation failed")
            return Result.failure(tts_engine_error(f"Gemini Async TTS failed: {e}"))

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

    def supports_ssml(self) -> bool:
        """Whether this engine supports SSML."""
        # Gemini takes a text prompt, not SSML. Delivery is steered by style_prompt instead.
        return False
