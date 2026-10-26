# tests/infrastructure/llm/test_gemini_llm_provider.py
"""Comprehensive unit tests for GeminiLLMProvider implementation.

Tests initialization, client management, content generation, async operations, and error handling.
"""

from unittest.mock import MagicMock, Mock, patch

from domain.errors import ErrorCode
from infrastructure.llm.gemini_llm_provider import GeminiLLMProvider


def assert_logged(log_method: Mock, expected: str) -> None:
    """Assert a mocked logger method was called with a message rendering to `expected`."""
    rendered = [c.args[0] % c.args[1:] if len(c.args) > 1 else c.args[0] for c in log_method.call_args_list]
    assert expected in rendered, f"{expected!r} not logged; got {rendered!r}"


class TestGeminiLLMProviderInitialization:
    """Test GeminiLLMProvider initialization and configuration."""

    def test_init_with_valid_api_key_creates_client(self) -> None:
        """Should initialize client with valid API key."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            assert provider.api_key == "valid_key"
            assert provider.model_name == "gemini-1.5-flash"
            assert provider.client == mock_client
            assert mock_client_class.call_args.kwargs["api_key"] == "valid_key"

    def test_init_with_placeholder_api_key_returns_none_client(self) -> None:
        """Should not create client with placeholder API key."""
        provider = GeminiLLMProvider(api_key="YOUR_GOOGLE_AI_API_KEY", model_name="gemini-1.5-flash")

        assert provider.api_key == "YOUR_GOOGLE_AI_API_KEY"
        assert provider.client is None

    def test_init_with_empty_api_key_returns_none_client(self) -> None:
        """Should not create client with empty API key."""
        provider = GeminiLLMProvider(api_key="", model_name="gemini-1.5-flash")

        assert provider.api_key == ""
        assert provider.client is None

    def test_init_with_none_api_key_returns_none_client(self) -> None:
        """Should not create client with None API key."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client"):
            provider = GeminiLLMProvider(
                api_key=None,  # type: ignore[arg-type]
                model_name="gemini-1.5-flash",
            )

            assert provider.api_key is None
            assert provider.client is None  # type: ignore[unreachable]

    def test_init_with_client_exception_returns_none_client(self) -> None:
        """Should return None client when genai.Client raises exception."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            mock_client_class.side_effect = Exception("API initialization failed")

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            assert provider.client is None


class TestGeminiLLMProviderClientManagement:
    """Test client initialization and management."""

    def test_init_client_with_valid_key_returns_client(self) -> None:
        """Should return client instance with valid API key."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            result = provider._init_client()

            assert result == mock_client
            assert mock_client_class.call_args.kwargs["api_key"] == "valid_key"

    def test_init_client_with_invalid_key_returns_none(self) -> None:
        """Should return None with invalid API key."""
        provider = GeminiLLMProvider(api_key="YOUR_GOOGLE_AI_API_KEY", model_name="gemini-1.5-flash")

        result = provider._init_client()

        assert result is None

    def test_init_client_with_exception_returns_none(self) -> None:
        """Should return None when client initialization raises exception."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            mock_client_class.side_effect = ValueError("Invalid API key format")

            provider = GeminiLLMProvider(api_key="malformed_key", model_name="gemini-1.5-flash")

            result = provider._init_client()

            assert result is None

    def test_methods_check_client_availability(self) -> None:
        """Should check client availability in generate_content method."""
        provider = GeminiLLMProvider(
            api_key="YOUR_GOOGLE_AI_API_KEY",  # Creates None client
            model_name="gemini-1.5-flash",
        )

        result = provider.generate_content("test prompt")

        assert result.is_failure
        assert result.error is not None
        assert result.error.code == ErrorCode.LLM_PROVIDER_ERROR
        assert result.error.details == "Gemini API key not configured"


class TestGeminiLLMProviderSyncContentGeneration:
    """Test synchronous content generation with various response scenarios."""

    def test_generate_content_success_with_complex_response_structure(self) -> None:
        """Should successfully parse complex response structure."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create mock response with complex structure
            mock_response = Mock()
            mock_candidate = Mock()
            mock_content = Mock()
            mock_part = Mock()
            mock_part.text = "Generated content from complex structure"
            mock_content.parts = [mock_part]
            mock_candidate.content = mock_content
            mock_candidate.finish_reason = "STOP"
            mock_response.candidates = [mock_candidate]

            # Setup mock client
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.5]),
                patch("infrastructure.llm.gemini_llm_provider.logger") as mock_logger,
            ):
                result = provider.generate_content("test prompt")

            assert result.is_success
            assert result.value == "Generated content from complex structure"

            # Verify API call
            mock_client.models.generate_content.assert_called_once()
            call_args = mock_client.models.generate_content.call_args
            assert call_args.kwargs["model"] == "gemini-1.5-flash"
            assert call_args.kwargs["contents"] == "test prompt"

            # Verify logging
            assert_logged(mock_logger.info, "LLM API Call: Model=gemini-1.5-flash, prompt='test prompt' (11 chars)")
            assert_logged(mock_logger.info, "LLM API Success: 1.50s for 'test prompt'")
            assert_logged(mock_logger.debug, "Response finish_reason: STOP")

    def test_generate_content_success_with_simple_text_accessor(self) -> None:
        """Should fallback to simple .text accessor when complex structure fails."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create mock response with simple .text accessor
            mock_response = Mock()
            mock_response.candidates = []  # Empty candidates to trigger fallback
            mock_response.text = "Generated content via text accessor"

            # Setup mock client
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger") as mock_logger,
            ):
                result = provider.generate_content("test prompt")

            assert result.is_success
            assert result.value == "Generated content via text accessor"

            # Verify fallback logging
            assert_logged(mock_logger.debug, "LLM Response: 35 chars returned (via .text accessor)")

    def test_generate_content_handles_empty_response(self) -> None:
        """Should handle empty response appropriately."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create empty mock response
            mock_response = Mock()
            mock_response.candidates = []
            mock_response.text = None

            # Setup mock client
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger") as mock_logger,
            ):
                result = provider.generate_content("test prompt")

            assert result.is_failure
            assert result.error is not None
            assert result.error.code == ErrorCode.LLM_PROVIDER_ERROR
            assert result.error.details == "Empty response from LLM"

            # Verify debug logging
            assert_logged(mock_logger.warning, "LLM API: No text in response for 'test prompt'")

    def test_generate_content_handles_no_candidates(self) -> None:
        """Should handle response with no candidates."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create mock response with no candidates
            mock_response = Mock()
            mock_response.candidates = None
            mock_response.text = None

            # Setup mock client
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger"),
            ):
                result = provider.generate_content("test prompt")

            assert result.is_failure
            assert result.error is not None
            assert result.error.code == ErrorCode.LLM_PROVIDER_ERROR
            assert result.error.details == "Empty response from LLM"

    def test_generate_content_handles_no_content_parts(self) -> None:
        """Should handle candidate with no content parts."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create mock response with candidate but no parts
            mock_response = Mock()
            mock_candidate = Mock()
            mock_content = Mock()
            mock_content.parts = []  # Empty parts
            mock_candidate.content = mock_content
            mock_candidate.finish_reason = "LENGTH"
            mock_response.candidates = [mock_candidate]
            mock_response.text = None

            # Setup mock client
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger") as mock_logger,
            ):
                result = provider.generate_content("test prompt")

            assert result.is_failure
            assert result.error is not None
            assert result.error.details == "Empty response from LLM"

            # Should still log finish_reason
            assert_logged(mock_logger.debug, "Response finish_reason: LENGTH")

    def test_generate_content_handles_api_exception(self) -> None:
        """Should handle exceptions during API call."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Setup mock client that raises exception
            mock_client = Mock()
            mock_client.models.generate_content.side_effect = Exception("API quota exceeded")
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.5]),
                patch("infrastructure.llm.gemini_llm_provider.logger") as mock_logger,
            ):
                result = provider.generate_content("test prompt")

            assert result.is_failure
            assert result.error is not None
            assert result.error.code == ErrorCode.LLM_PROVIDER_ERROR
            assert result.error.details == "Content generation failed: API quota exceeded"

            # Verify error logging
            assert_logged(mock_logger.error, "LLM API Error (1.50s): API quota exceeded for 'test prompt'")

    def test_generate_content_truncates_long_prompts_in_logs(self) -> None:
        """Should truncate long prompts in log messages."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create successful mock response
            mock_response = Mock()
            mock_response.text = "Generated content"
            mock_response.candidates = []

            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            long_prompt = "This is a very long prompt " * 10  # > 100 chars

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger") as mock_logger,
            ):
                result = provider.generate_content(long_prompt)

            assert result.is_success

            # Verify prompt truncation in logs
            expected_preview = long_prompt[:100] + "..."
            expected_call = (
                f"LLM API Call: Model=gemini-1.5-flash, prompt='{expected_preview}' ({len(long_prompt)} chars)"
            )
            assert_logged(mock_logger.info, expected_call)

    def test_generate_content_handles_response_inspection_error(self) -> None:
        """Should handle errors during response structure inspection."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create mock response that raises exception during inspection
            mock_response = Mock()
            # Mock candidates as a list-like object that raises exception when accessed
            mock_candidates = MagicMock()
            mock_candidates.__getitem__.side_effect = Exception("Inspection failed")
            mock_response.candidates = mock_candidates
            mock_response.text = "Fallback content"

            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger") as mock_logger,
            ):
                result = provider.generate_content("test prompt")

            assert result.is_success
            assert result.value == "Fallback content"

            # Should log inspection error
            assert_logged(mock_logger.warning, "Error inspecting response structure: Inspection failed")

    def test_generate_content_logs_response_debugging_info(self) -> None:
        """Should log response debugging information for empty responses."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create empty mock response with debugging info
            mock_response = Mock()
            mock_response.candidates = []
            mock_response.text = None

            # Mock response type and attributes for debugging
            type(mock_response).__name__ = "GenerateContentResponse"
            _dir_attrs_407 = [
                "candidates",
                "text",
                "prompt_feedback",
                "usage_metadata",
                "model",
                "finish_reason",
                "safety_ratings",
                "citation_metadata",
                "content_filter_results",
                "blocked",
            ]
            type(mock_response).__dir__ = lambda self: _dir_attrs_407  # type: ignore[method-assign]

            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger") as mock_logger,
            ):
                result = provider.generate_content("test prompt")

            assert result.is_failure

            # Verify debugging logs
            assert_logged(mock_logger.debug, "Response object type: <class 'unittest.mock.Mock'>")
            expected_attrs = (
                "['blocked', 'candidates', 'citation_metadata', 'content_filter_results', "
                "'finish_reason', 'model', 'prompt_feedback', 'safety_ratings', 'text', 'usage_metadata']..."
            )
            assert_logged(mock_logger.debug, f"Response attributes: {expected_attrs}")

    def test_generate_content_with_valid_content_but_empty_text(self) -> None:
        """Should handle response with valid structure but empty text content."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create mock response with valid structure but empty text
            mock_response = Mock()
            mock_candidate = Mock()
            mock_content = Mock()
            mock_part = Mock()
            mock_part.text = ""  # Empty text
            mock_content.parts = [mock_part]
            mock_candidate.content = mock_content
            mock_candidate.finish_reason = "STOP"
            mock_response.candidates = [mock_candidate]
            mock_response.text = None

            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger"),
            ):
                result = provider.generate_content("test prompt")

            assert result.is_failure
            assert result.error is not None
            assert result.error.details == "Empty response from LLM"

    def test_generate_content_uses_correct_api_configuration(self) -> None:
        """Should use correct API configuration for generate_content call."""
        with (
            patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class,
            patch("infrastructure.llm.gemini_llm_provider.types") as mock_types,
        ):
            # Create mock response
            mock_response = Mock()
            mock_response.text = "Generated content"
            mock_response.candidates = []

            # Setup mock client
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Setup mock config
            mock_config = Mock()
            mock_types.GenerateContentConfig.return_value = mock_config

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-pro")

            with patch("time.time", side_effect=[1000.0, 1001.0]):
                result = provider.generate_content("test prompt")

            assert result.is_success

            # Verify API call parameters
            mock_client.models.generate_content.assert_called_once_with(
                model="gemini-1.5-pro", contents="test prompt", config=mock_config
            )

            # Verify config creation
            mock_types.GenerateContentConfig.assert_called_once_with(max_output_tokens=30000, temperature=0.3)


class TestGeminiLLMProviderResponseParsing:
    """Test response parsing logic and fallback mechanisms."""

    def test_response_parsing_with_nested_structure_success(self) -> None:
        """Should successfully parse deeply nested response structure."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create deeply nested mock response
            mock_response = Mock()
            mock_candidate = Mock()
            mock_content = Mock()
            mock_part1 = Mock()
            mock_part1.text = "First part content"
            mock_part2 = Mock()
            mock_part2.text = "Second part content"
            mock_content.parts = [mock_part1, mock_part2]
            mock_candidate.content = mock_content
            mock_candidate.finish_reason = "STOP"
            mock_response.candidates = [mock_candidate]

            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger"),
            ):
                result = provider.generate_content("test prompt")

            assert result.is_success
            # Should use first part's text
            assert result.value == "First part content"

    def test_response_parsing_handles_attribute_inspection(self) -> None:
        """Should handle response attribute inspection for debugging."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create mock response that will fail parsing and trigger inspection
            mock_response = Mock()
            mock_response.candidates = []
            mock_response.text = None
            mock_response.__class__.__name__ = "GenerateContentResponse"

            # Mock dir() to return specific attributes
            _dir_attrs_736 = [
                "candidates",
                "text",
                "finish_reason",
                "usage_metadata",
                "prompt_feedback",
                "safety_ratings",
                "citation_metadata",
                "model_name",
                "generation_config",
                "safety_settings",
            ]
            type(mock_response).__dir__ = lambda self: _dir_attrs_736  # type: ignore[method-assign]

            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger") as mock_logger,
            ):
                result = provider.generate_content("test prompt")

            assert result.is_failure

            # Verify attribute inspection logging
            assert_logged(mock_logger.debug, "Response object type: <class 'unittest.mock.Mock'>")
            # Should show first 10 attributes (sorted alphabetically by dir())
            expected_attrs = (
                "['candidates', 'citation_metadata', 'finish_reason', 'generation_config', "
                "'model_name', 'prompt_feedback', 'safety_ratings', 'safety_settings', 'text', 'usage_metadata']..."
            )
            assert_logged(mock_logger.debug, f"Response attributes: {expected_attrs}")

    def test_response_parsing_multiple_fallback_paths(self) -> None:
        """Should test multiple fallback paths in response parsing."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client") as mock_client_class:
            # Create mock response that will trigger all parsing paths
            mock_response = Mock()

            # First, it will try candidates path (fail)
            mock_response.candidates = None

            # Then it will try simple .text accessor (succeed)
            mock_response.text = "Fallback text content"

            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            provider = GeminiLLMProvider(api_key="valid_key", model_name="gemini-1.5-flash")

            with (
                patch("time.time", side_effect=[1000.0, 1001.0]),
                patch("infrastructure.llm.gemini_llm_provider.logger") as mock_logger,
            ):
                result = provider.generate_content("test prompt")

            assert result.is_success
            assert result.value == "Fallback text content"

            # Should log the fallback usage
            assert_logged(mock_logger.debug, "LLM Response: 21 chars returned (via .text accessor)")


class TestGeminiLLMProviderIntegration:
    """Test integration functionality and wrapper methods."""
