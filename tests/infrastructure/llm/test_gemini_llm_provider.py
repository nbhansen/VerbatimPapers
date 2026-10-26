# tests/infrastructure/llm/test_gemini_llm_provider.py
"""GeminiLLMProvider: prompt in, the model's full text out, failures as Result.

Responses are real SDK objects rather than Mocks, so parsing is tested against
the shape Gemini actually returns.
"""

import logging
from unittest.mock import Mock, patch

from google.genai import types
import pytest

from domain.errors import ErrorCode
from infrastructure.llm.gemini_llm_provider import GeminiLLMProvider


def _response(
    *parts: types.Part, finish_reason: types.FinishReason = types.FinishReason.STOP
) -> types.GenerateContentResponse:
    """A response with one candidate holding the given parts."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=list(parts)), finish_reason=finish_reason)
        ]
    )


def _provider(outcome: types.GenerateContentResponse | Exception) -> tuple[GeminiLLMProvider, Mock]:
    """Provider whose client returns (or raises) the given outcome."""
    client = Mock()
    if isinstance(outcome, Exception):
        client.models.generate_content.side_effect = outcome
    else:
        client.models.generate_content.return_value = outcome
    with patch("infrastructure.llm.gemini_llm_provider.genai.Client", return_value=client):
        return GeminiLLMProvider(api_key="valid_key", model_name="gemini-text"), client


class TestWithoutUsableKey:
    """No client is built without a real key, and calls fail with a clear reason."""

    @pytest.mark.parametrize("api_key", ["", None, "YOUR_GOOGLE_AI_API_KEY"])
    def test_missing_or_placeholder_key(self, api_key: str | None) -> None:
        """Every call fails, saying the key isn't configured."""
        provider = GeminiLLMProvider(api_key=api_key, model_name="gemini-text")  # type: ignore[arg-type]

        result = provider.generate_content("prompt")

        assert provider.client is None
        assert result.is_failure and result.error is not None
        assert result.error.code == ErrorCode.LLM_PROVIDER_ERROR
        assert result.error.details == "Gemini API key not configured"

    def test_client_construction_failure_is_deferred(self) -> None:
        """A client that can't be built fails calls instead of crashing startup."""
        with patch("infrastructure.llm.gemini_llm_provider.genai.Client", side_effect=ValueError("bad key format")):
            provider = GeminiLLMProvider(api_key="malformed", model_name="gemini-text")

        result = provider.generate_content("prompt")

        assert result.is_failure and result.error is not None
        assert "bad key format" in (result.error.details or "")


class TestGenerateContent:
    """The model's whole reply comes back; anything less is a failure."""

    def test_returns_the_text_and_sends_the_prompt(self) -> None:
        """The prompt goes to the configured text model, and its reply is returned."""
        provider, client = _provider(_response(types.Part(text="Cleaned text.")))

        result = provider.generate_content("Clean this.")

        assert result.value == "Cleaned text."
        kwargs = client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-text"
        assert kwargs["contents"] == "Clean this."

    def test_reply_split_across_parts_is_joined(self) -> None:
        """A long reply can arrive in several parts; dropping any of them loses document text."""
        provider, _ = _provider(_response(types.Part(text="First half, "), types.Part(text="second half.")))

        assert provider.generate_content("Clean this.").value == "First half, second half."

    def test_thought_parts_are_not_returned(self) -> None:
        """A thinking model's reasoning must never be read aloud as part of the paper."""
        provider, _ = _provider(
            _response(types.Part(text="Let me remove the headers...", thought=True), types.Part(text="Cleaned text."))
        )

        assert provider.generate_content("Clean this.").value == "Cleaned text."

    def test_truncated_reply_is_a_failure(self) -> None:
        """Output cut off at the token limit would drop the rest of the passage, so it isn't used."""
        provider, _ = _provider(
            _response(types.Part(text="The first half of the cleaned"), finish_reason=types.FinishReason.MAX_TOKENS)
        )

        result = provider.generate_content("Clean this.")

        assert result.is_failure and result.error is not None
        assert "truncated" in (result.error.details or "")

    @pytest.mark.parametrize(
        "response",
        [
            types.GenerateContentResponse(candidates=[]),
            _response(),
            _response(types.Part(text="Reasoning only.", thought=True)),
        ],
        ids=["no-candidates", "no-parts", "only-thoughts"],
    )
    def test_no_text_is_a_failure(self, response: types.GenerateContentResponse) -> None:
        """An empty reply fails, so the pipeline falls back instead of speaking nothing."""
        provider, _ = _provider(response)

        result = provider.generate_content("Clean this.")

        assert result.is_failure and result.error is not None
        assert result.error.details == "Empty response from LLM"

    def test_api_exception_is_a_failure(self) -> None:
        """An API error is returned with its reason, not raised."""
        provider, _ = _provider(Exception("quota exceeded"))

        result = provider.generate_content("Clean this.")

        assert result.is_failure and result.error is not None
        assert "quota exceeded" in (result.error.details or "")


class TestLoggingKeepsDocumentsPrivate:
    """Prompts are the user's document text; the log (~/.verbatimpapers/app.log) must never hold it."""

    DOCUMENT_TEXT = "Confidential manuscript sentence about zebrafish neurons."

    @pytest.mark.parametrize(
        "outcome",
        [
            _response(types.Part(text="Cleaned output.")),
            _response(),
            _response(types.Part(text="Cut"), finish_reason=types.FinishReason.MAX_TOKENS),
            Exception("quota exceeded"),
        ],
        ids=["success", "empty-response", "truncated", "api-error"],
    )
    def test_prompt_text_never_logged(
        self, outcome: types.GenerateContentResponse | Exception, caplog: pytest.LogCaptureFixture
    ) -> None:
        """At every level, on every path, only the prompt's length is logged."""
        provider, _ = _provider(outcome)

        with caplog.at_level(logging.DEBUG, logger="infrastructure.llm.gemini_llm_provider"):
            provider.generate_content(f"Clean this:\n{self.DOCUMENT_TEXT}")

        assert caplog.records, "The call should be logged"
        assert "zebrafish" not in caplog.text

    def test_api_failure_is_logged_as_an_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failed call leaves an ERROR line with the API's reason, so it can be diagnosed."""
        provider, _ = _provider(Exception("quota exceeded"))

        with caplog.at_level(logging.DEBUG, logger="infrastructure.llm.gemini_llm_provider"):
            provider.generate_content(self.DOCUMENT_TEXT)

        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1 and "quota exceeded" in errors[0]
