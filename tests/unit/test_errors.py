"""Result and the error helpers: what the rest of the app relies on them to carry."""

import pytest

from application.services.document_service import _GENERIC_MESSAGES
from domain.errors import (
    ApplicationError,
    ErrorCode,
    Result,
    audio_generation_error,
    configuration_error,
    invalid_page_range_error,
    llm_provider_error,
    text_extraction_error,
    tts_engine_error,
)


def test_from_exception_keeps_the_text_and_names_the_type() -> None:
    """The exception's text is the message and its class the details, so neither is lost."""
    result: Result[None] = Result.from_exception(ConnectionError("Network timeout"), ErrorCode.TTS_ENGINE_ERROR, True)

    assert result.is_failure and not result.is_success
    assert result.error == ApplicationError(
        code=ErrorCode.TTS_ENGINE_ERROR, message="Network timeout", details="ConnectionError", retryable=True
    )


@pytest.mark.parametrize(
    ("helper", "code"),
    [
        (text_extraction_error, ErrorCode.TEXT_EXTRACTION_FAILED),
        (invalid_page_range_error, ErrorCode.INVALID_PAGE_RANGE),
        (audio_generation_error, ErrorCode.AUDIO_GENERATION_FAILED),
        (tts_engine_error, ErrorCode.TTS_ENGINE_ERROR),
        (llm_provider_error, ErrorCode.LLM_PROVIDER_ERROR),
        (configuration_error, ErrorCode.CONFIGURATION_ERROR),
    ],
)
def test_helper_messages_are_known_to_failure_descriptions(helper: object, code: ErrorCode) -> None:
    """describe_failure drops a helper's fixed message in favour of its details; renaming one breaks that."""
    error = helper("why it failed")  # type: ignore[operator]

    assert error.code == code
    assert error.details == "why it failed"
    assert error.message in _GENERIC_MESSAGES
