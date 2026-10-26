# domain/errors.py
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar


class ErrorCode(Enum):
    """Standard error codes for the application."""

    TEXT_EXTRACTION_FAILED = "text_extraction_failed"
    TEXT_CLEANING_FAILED = "text_cleaning_failed"
    AUDIO_GENERATION_FAILED = "audio_generation_failed"
    INVALID_PAGE_RANGE = "invalid_page_range"
    CONFIGURATION_ERROR = "configuration_error"
    TTS_ENGINE_ERROR = "tts_engine_error"
    LLM_PROVIDER_ERROR = "llm_provider_error"
    CANCELLED = "cancelled"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ApplicationError:
    """Structured error information."""

    code: ErrorCode
    message: str
    details: str | None = None
    retryable: bool = False

    def __str__(self) -> str:
        """Return string representation of the error."""
        return f"{self.code.value}: {self.message}"

    def __repr__(self) -> str:
        """Return detailed string representation for debugging."""
        return f"ApplicationError(code={self.code.value}, message='{self.message}', retryable={self.retryable})"


# Result type for better error handling
T = TypeVar("T")


@dataclass
class Result[T]:
    """Result type that either contains a value or an error."""

    value: T | None = None
    error: ApplicationError | None = None

    @property
    def is_success(self) -> bool:
        """Return True if the result contains a successful value."""
        return self.error is None

    @property
    def is_failure(self) -> bool:
        """Return True if the result contains an error."""
        return self.error is not None

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """Create a successful result with the given value."""
        return cls(value=value)

    @classmethod
    def failure(cls, error: ApplicationError) -> "Result[T]":
        """Create a failed result with the given error."""
        return cls(error=error)

    @classmethod
    def from_exception(
        cls, ex: Exception, code: ErrorCode = ErrorCode.UNKNOWN_ERROR, retryable: bool = False
    ) -> "Result[T]":
        """Create a failed result from an exception."""
        return cls.failure(
            ApplicationError(code=code, message=str(ex), details=ex.__class__.__name__, retryable=retryable)
        )


# Helper functions for common error types


def text_extraction_error(details: str) -> ApplicationError:
    """Create an error for text extraction failures."""
    return ApplicationError(
        code=ErrorCode.TEXT_EXTRACTION_FAILED,
        message="Failed to extract text from PDF",
        details=details,
        retryable=True,  # Text extraction errors are often transient (OCR, file access)
    )


def invalid_page_range_error(details: str) -> ApplicationError:
    """Create an error for a page range that doesn't fit the document."""
    return ApplicationError(
        code=ErrorCode.INVALID_PAGE_RANGE, message="Invalid page range", details=details, retryable=False
    )


def audio_generation_error(details: str | None = None) -> ApplicationError:
    """Create an error for audio generation failures."""
    return ApplicationError(
        code=ErrorCode.AUDIO_GENERATION_FAILED,
        message="Audio generation failed",
        details=details,
        retryable=True,  # TTS errors might be transient
    )


def tts_engine_error(details: str | None = None) -> ApplicationError:
    """Create an error for TTS engine failures."""
    return ApplicationError(
        code=ErrorCode.TTS_ENGINE_ERROR,
        message="TTS engine error",
        details=details,
        retryable=True,  # Rate limits, API issues, etc.
    )


def llm_provider_error(details: str | None = None) -> ApplicationError:
    """Create an error for LLM provider failures."""
    return ApplicationError(
        code=ErrorCode.LLM_PROVIDER_ERROR,
        message="LLM provider error",
        details=details,
        retryable=True,  # API issues, rate limits, etc.
    )


def cancelled_error() -> ApplicationError:
    """Create the error returned when the user cancelled the operation."""
    return ApplicationError(code=ErrorCode.CANCELLED, message="Cancelled by user", retryable=False)


def configuration_error(details: str) -> ApplicationError:
    """Create an error for configuration problems."""
    return ApplicationError(
        code=ErrorCode.CONFIGURATION_ERROR, message="Configuration error", details=details, retryable=False
    )
