"""Infrastructure tests for GeminiTTSProvider.

Tests the Gemini TTS provider implementation and interface compliance.
The Gemini API client is mocked; these tests cover request configuration,
response parsing (including PCM-to-WAV conversion), and error handling.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from domain.errors import ErrorCode
from infrastructure.tts.gemini_tts_provider import DEFAULT_STYLE_PROMPT, GeminiTTSProvider


def make_audio_response(data: bytes, mime_type: str = "audio/mp3") -> SimpleNamespace:
    """Build a fake Gemini response containing one inline audio part."""
    part = SimpleNamespace(inline_data=SimpleNamespace(mime_type=mime_type, data=data))
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    return SimpleNamespace(candidates=[candidate])


@pytest.fixture
def basic_gemini_provider() -> GeminiTTSProvider:
    """Basic GeminiTTSProvider for testing."""
    return GeminiTTSProvider(
        model_name="gemini-1.5-flash",
        api_key="test_api_key",
        voice_name="Kore",
    )


@pytest.fixture
def custom_gemini_provider() -> GeminiTTSProvider:
    """Custom GeminiTTSProvider with specific settings."""
    return GeminiTTSProvider(
        model_name="gemini-1.5-pro",
        api_key="custom_api_key",
        voice_name="Aria",
        min_request_interval=1.5,
        max_concurrent_requests=8,
        requests_per_minute=60,
    )


class TestGeminiTTSProviderInitialization:
    """Test GeminiTTSProvider initialization and configuration."""

    def test_init_with_basic_parameters(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Should initialize with basic parameters."""
        provider = basic_gemini_provider

        assert provider.model_name == "gemini-1.5-flash"
        assert provider.api_key == "test_api_key"
        assert provider.voice_name == "Kore"

        # Default values
        assert provider.min_request_interval == 2.0
        assert provider.max_concurrent_requests == 4
        assert provider.requests_per_minute == 30

    def test_init_with_custom_parameters(self, custom_gemini_provider: GeminiTTSProvider) -> None:
        """Should initialize with custom parameters."""
        provider = custom_gemini_provider

        assert provider.model_name == "gemini-1.5-pro"
        assert provider.api_key == "custom_api_key"
        assert provider.voice_name == "Aria"
        assert provider.min_request_interval == 1.5
        assert provider.max_concurrent_requests == 8
        assert provider.requests_per_minute == 60

    def test_init_client_failure_is_deferred(self) -> None:
        """Client construction errors should be stored, not raised."""
        with patch("infrastructure.tts.gemini_tts_provider.genai.Client") as mock_client_class:
            mock_client_class.side_effect = ValueError("bad key")

            provider = GeminiTTSProvider(model_name="gemini-1.5-flash", api_key="bad", voice_name="Kore")
            result = provider.synthesize("Hello")

            assert result.is_failure
            assert result.error is not None
            assert result.error.code == ErrorCode.TTS_ENGINE_ERROR
            assert "Failed to initialize Gemini Client" in str(result.error.details)

            # The same error is reported by the startup readiness check
            ready = provider.check_ready()
            assert ready.is_failure
            assert "Failed to initialize Gemini Client" in str(ready.error.details if ready.error else "")

    def test_ready_once_client_is_built(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """A built client is ready; the service itself is not contacted."""
        assert basic_gemini_provider.check_ready().is_success


class TestGeminiTTSProviderInterfaces:
    """Test ITTSEngine interface implementation."""

    def test_supports_ssml(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Gemini TTS takes plain text prompts, not SSML."""
        assert basic_gemini_provider.supports_ssml() is False


class TestGeminiTTSProviderTextValidation:
    """Test text input validation for TTS generation."""

    def test_synthesize_rejects_empty_text(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Should reject empty text input."""
        result = basic_gemini_provider.synthesize("")

        assert result.is_failure
        assert result.error is not None
        assert result.error.code == ErrorCode.TTS_ENGINE_ERROR
        assert "Empty text provided" in str(result.error.details)

    def test_synthesize_rejects_whitespace_only(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Should reject whitespace-only text."""
        result = basic_gemini_provider.synthesize("   \n\t  ")

        assert result.is_failure
        assert result.error is not None
        assert result.error.code == ErrorCode.TTS_ENGINE_ERROR
        assert "Empty text provided" in str(result.error.details)

    def test_empty_text_error_consistency(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Should consistently reject all empty inputs."""
        for empty_input in ["", "   ", "\n\n", "\t\t", "  \n  \t  "]:
            result = basic_gemini_provider.synthesize(empty_input)
            assert result.is_failure
            assert result.error is not None
            assert result.error.code == ErrorCode.TTS_ENGINE_ERROR
            assert "Empty text provided" in str(result.error.details)


class TestGeminiTTSProviderSyncGeneration:
    """Test synchronous TTS generation with a mocked client."""

    def test_synthesize_success(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Should return audio bytes from the API response."""
        provider = basic_gemini_provider
        mock_client = Mock()
        mock_client.models.generate_content.return_value = make_audio_response(
            b"\x00\x01" * 100, mime_type="audio/pcm;rate=24000"
        )
        provider.client = mock_client

        result = provider.synthesize("Hello, this is a test.")

        assert result.is_success
        assert result.value is not None
        # Gemini has no sentence structure, so the whole chunk is one segment
        assert len(result.value) == 1
        assert result.value[0].text == "Hello, this is a test."
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-1.5-flash"
        # Text is sent with the delivery directive prefixed - Gemini's only prosody control
        assert call_kwargs["contents"] == f"{DEFAULT_STYLE_PROMPT}\n\nHello, this is a test."

    def test_synthesize_handles_unicode(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Should pass Unicode text through to the API."""
        provider = basic_gemini_provider
        mock_client = Mock()
        mock_client.models.generate_content.return_value = make_audio_response(
            b"\x00\x01" * 100, mime_type="audio/pcm;rate=24000"
        )
        provider.client = mock_client

        unicode_text = "Héllo wörld! 你好世界 🌍"
        result = provider.synthesize(unicode_text)

        assert result.is_success
        assert mock_client.models.generate_content.call_args.kwargs["contents"].endswith(unicode_text)

    def test_pcm_returned_as_raw_samples(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """PCM stays raw - wrapping it in a container is AudioAssembler's job now."""
        provider = basic_gemini_provider
        mock_client = Mock()
        pcm = b"\x00\x01" * 100
        mock_client.models.generate_content.return_value = make_audio_response(pcm, mime_type="audio/pcm;rate=24000")
        provider.client = mock_client

        result = provider.synthesize("Hello")

        assert result.is_success
        assert result.value is not None
        segment = result.value[0]
        assert segment.pcm == pcm
        assert segment.sample_rate == 24000
        assert segment.sample_width == 2
        assert segment.channels == 1

    def test_synthesize_handles_api_exception(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """API exceptions should become failure Results, not raise."""
        provider = basic_gemini_provider
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
        provider.client = mock_client

        result = provider.synthesize("Hello")

        assert result.is_failure
        assert result.error is not None
        assert "quota exceeded" in str(result.error.details)


class TestGeminiTTSProviderAsyncGeneration:
    """Test async TTS generation with a mocked client."""

    @pytest.mark.asyncio
    async def test_synthesize_async_success(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Should generate audio asynchronously via the aio client."""
        provider = basic_gemini_provider
        mock_client = Mock()

        pcm = b"\x00\x01" * 100

        async def fake_generate(**_kwargs: object) -> SimpleNamespace:
            return make_audio_response(pcm, mime_type="audio/pcm;rate=24000")

        mock_client.aio.models.generate_content = fake_generate
        provider.client = mock_client

        result = await provider.synthesize_async("Test text for async generation")

        assert result.is_success
        assert result.value is not None
        assert result.value[0].pcm == pcm
        assert result.value[0].text == "Test text for async generation"

    @pytest.mark.asyncio
    async def test_synthesize_async_handles_api_exception(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Async API exceptions should become failure Results."""
        provider = basic_gemini_provider
        mock_client = Mock()

        async def fake_generate(**_kwargs: object) -> SimpleNamespace:
            raise RuntimeError("connection reset")

        mock_client.aio.models.generate_content = fake_generate
        provider.client = mock_client

        result = await provider.synthesize_async("Test text")

        assert result.is_failure
        assert result.error is not None
        assert "connection reset" in str(result.error.details)


class TestGeminiTTSProviderResponseParsing:
    """Test extraction of audio data from Gemini responses."""

    def test_no_candidates_is_failure(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Empty candidate list should be a failure."""
        empty_response: Any = SimpleNamespace(candidates=[])
        result = basic_gemini_provider._segments_from_response("Hello", empty_response)

        assert result.is_failure
        assert result.error is not None
        assert "No candidates" in str(result.error.details)

    def test_empty_content_is_failure(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Candidate without content parts should be a failure."""
        response: Any = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])
        result = basic_gemini_provider._segments_from_response("Hello", response)

        assert result.is_failure
        assert result.error is not None
        assert "Empty content" in str(result.error.details)

    def test_non_audio_parts_is_failure(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Parts without audio data should be a failure."""
        part = SimpleNamespace(inline_data=SimpleNamespace(mime_type="text/plain", data=b"not audio"))
        response: Any = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
        result = basic_gemini_provider._segments_from_response("Hello", response)

        assert result.is_failure
        assert result.error is not None
        assert "No audio data" in str(result.error.details)

    def test_pcm_sample_rate_parsed_from_mime_type(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """Sample rate in the PCM mime type should reach the segment."""
        response: Any = make_audio_response(b"\x00\x01" * 400, mime_type="audio/pcm;rate=16000")
        result = basic_gemini_provider._segments_from_response("Hello", response)

        assert result.is_success
        assert result.value is not None
        assert result.value[0].sample_rate == 16000

    def test_missing_sample_rate_falls_back_to_default(self, basic_gemini_provider: GeminiTTSProvider) -> None:
        """A PCM mime type with no rate should not crash."""
        response: Any = make_audio_response(b"\x00\x01" * 400, mime_type="audio/pcm")
        result = basic_gemini_provider._segments_from_response("Hello", response)

        assert result.is_success
        assert result.value is not None
        assert result.value[0].sample_rate == 24000


class TestGeminiTTSProviderStylePrompt:
    """Test the style prompt, which is how Gemini takes prosody direction.

    Gemini has no SSML; delivery is steered by a natural-language directive
    prefixed to the text. See issue #61.
    """

    @staticmethod
    def _provider(style_prompt: str | None) -> GeminiTTSProvider:
        return GeminiTTSProvider(
            model_name="gemini-1.5-flash",
            api_key="test_key",
            voice_name="Kore",
            style_prompt=style_prompt,
        )

    def test_defaults_to_builtin_style_prompt(self) -> None:
        """Omitting style_prompt should apply the academic-reading default."""
        assert self._provider(None).style_prompt == DEFAULT_STYLE_PROMPT

    def test_custom_style_prompt_is_used_verbatim(self) -> None:
        """A configured style prompt should replace the default."""
        provider = self._provider("Read this slowly.")

        assert provider.style_prompt == "Read this slowly."
        assert provider._build_prompt("Body text.") == "Read this slowly.\n\nBody text."

    def test_empty_style_prompt_sends_text_bare(self) -> None:
        """An empty string is the opt-out - no directive is prefixed."""
        provider = self._provider("")

        assert provider._build_prompt("Body text.") == "Body text."

    @pytest.mark.asyncio
    async def test_style_prompt_applied_on_async_path(self) -> None:
        """The async path must prefix the directive too, not just the sync one."""
        provider = self._provider("Read this slowly.")
        mock_client = Mock()

        async def fake_generate(**kwargs: object) -> SimpleNamespace:
            return make_audio_response(b"\x00\x01" * 100, mime_type="audio/pcm;rate=24000")

        mock_client.aio.models.generate_content = Mock(side_effect=fake_generate)
        provider.client = mock_client

        result = await provider.synthesize_async("Body text.")

        assert result.is_success
        contents = mock_client.aio.models.generate_content.call_args.kwargs["contents"]
        assert contents == "Read this slowly.\n\nBody text."


class TestGeminiTTSProviderConfiguration:
    """Test provider configuration and settings."""

    def test_voice_name_configuration(self) -> None:
        """Should accept different voice names."""
        for voice_name in ["Kore", "Aria", "Nova", "Sage"]:
            provider = GeminiTTSProvider(
                model_name="gemini-1.5-flash",
                api_key="test_key",
                voice_name=voice_name,
            )
            assert provider.voice_name == voice_name

    def test_model_name_configuration(self) -> None:
        """Should accept different model names."""
        for model_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]:
            provider = GeminiTTSProvider(
                model_name=model_name,
                api_key="test_key",
                voice_name="Kore",
            )
            assert provider.model_name == model_name

    def test_rate_limiting_configuration(self) -> None:
        """Should accept rate limiting configuration."""
        provider = GeminiTTSProvider(
            model_name="gemini-1.5-flash",
            api_key="test_key",
            voice_name="Kore",
            min_request_interval=0.5,
            max_concurrent_requests=10,
            requests_per_minute=120,
        )

        assert provider.min_request_interval == 0.5
        assert provider.max_concurrent_requests == 10
        assert provider.requests_per_minute == 120
