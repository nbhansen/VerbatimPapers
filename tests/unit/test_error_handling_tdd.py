# tests/unit/test_error_handling_tdd.py
"""TDD tests for Error Handling System - comprehensive coverage following red-green-refactor cycle.

Tests error creation, result types, error classification, and error factory functions.
"""

from domain.errors import (
    ApplicationError,
    ErrorCode,
    Result,
    audio_generation_error,
    configuration_error,
    llm_provider_error,
    text_extraction_error,
    tts_engine_error,
)


class TestResultType:
    """TDD tests for Result[T] generic type."""

    def test_result_success_creation(self):
        """Should create successful result with value."""
        result = Result.success("test_value")

        assert result.value == "test_value"
        assert result.error is None
        assert result.is_success is True
        assert result.is_failure is False

    def test_result_failure_creation(self):
        """Should create failed result with error."""
        error = ApplicationError(ErrorCode.UNKNOWN_ERROR, "Test error")
        result: Result[None] = Result.failure(error)

        assert result.value is None
        assert result.error == error
        assert result.is_success is False
        assert result.is_failure is True

    def test_result_success_with_different_types(self):
        """Should support different value types."""
        # String result
        str_result = Result.success("hello")
        assert str_result.value == "hello"
        assert isinstance(str_result.value, str)

        # Integer result
        int_result = Result.success(42)
        assert int_result.value == 42
        assert isinstance(int_result.value, int)

        # List result
        list_result = Result.success([1, 2, 3])
        assert list_result.value == [1, 2, 3]
        assert isinstance(str_result.value, str)

        # None result (valid case)
        none_result = Result.success(None)
        assert none_result.value is None
        assert none_result.is_success is True

    def test_result_from_exception_with_defaults(self):
        """Should create result from exception with default error code."""
        exception = ValueError("Invalid input")
        result: Result[None] = Result.from_exception(exception)

        assert result.is_failure is True
        assert result.error is not None
        assert result.error.code == ErrorCode.UNKNOWN_ERROR
        assert result.error.message == "Invalid input"
        assert result.error.details == "ValueError"
        assert result.error.retryable is False

    def test_result_from_exception_with_custom_args(self):
        """Should create result from exception with custom error code and retryable flag."""
        exception = ConnectionError("Network timeout")
        result: Result[None] = Result.from_exception(exception, code=ErrorCode.TTS_ENGINE_ERROR, retryable=True)

        assert result.is_failure is True
        assert result.error is not None
        assert result.error.code == ErrorCode.TTS_ENGINE_ERROR
        assert result.error.message == "Network timeout"
        assert result.error.details == "ConnectionError"
        assert result.error.retryable is True

    def test_result_from_exception_with_complex_exception(self):
        """Should handle complex exception types."""

        class CustomException(Exception):
            def __init__(self, message: str, error_code: int) -> None:
                super().__init__(message)
                self.error_code = error_code

        exception = CustomException("Custom error occurred", 500)
        result: Result[None] = Result.from_exception(exception, ErrorCode.CONFIGURATION_ERROR)

        assert result.error is not None
        assert result.error.message == "Custom error occurred"
        assert result.error.details == "CustomException"
        assert result.error.code == ErrorCode.CONFIGURATION_ERROR

    def test_result_mutually_exclusive_success_failure(self):
        """Should ensure success and failure are mutually exclusive."""
        success_result = Result.success("value")
        failure_result: Result[None] = Result.failure(ApplicationError(ErrorCode.UNKNOWN_ERROR, "error"))

        # Success result
        assert success_result.is_success
        assert not success_result.is_failure

        # Failure result
        assert failure_result.is_failure
        assert not failure_result.is_success

    def test_result_type_hints_work_correctly(self):
        """Should work correctly with type annotations."""

        # This test documents that the generic typing works
        def process_data() -> Result[str]:
            return Result.success("processed")

        def process_number() -> Result[int]:
            return Result.success(42)

        str_result = process_data()
        int_result = process_number()

        assert isinstance(str_result.value, str)
        assert isinstance(int_result.value, int)


class TestErrorFactoryFunctions:
    """TDD tests for error factory functions."""

    def test_text_extraction_error_factory(self):
        """Should create text extraction error with correct properties."""
        error = text_extraction_error("PDF is corrupted")

        assert error.code == ErrorCode.TEXT_EXTRACTION_FAILED
        assert error.message == "Failed to extract text from PDF"
        assert error.details == "PDF is corrupted"
        assert error.retryable is True  # Text extraction errors should be retryable

    def test_audio_generation_error_factory_with_details(self):
        """Should create audio generation error with details."""
        error = audio_generation_error("Model not available")

        assert error.code == ErrorCode.AUDIO_GENERATION_FAILED
        assert error.message == "Audio generation failed"
        assert error.details == "Model not available"
        assert error.retryable is True

    def test_audio_generation_error_factory_without_details(self):
        """Should create audio generation error without details."""
        error = audio_generation_error()

        assert error.code == ErrorCode.AUDIO_GENERATION_FAILED
        assert error.message == "Audio generation failed"
        assert error.details is None
        assert error.retryable is True

    def test_tts_engine_error_factory(self):
        """Should create TTS engine error with correct properties."""
        error = tts_engine_error("Rate limit exceeded")

        assert error.code == ErrorCode.TTS_ENGINE_ERROR
        assert error.message == "TTS engine error"
        assert error.details == "Rate limit exceeded"
        assert error.retryable is True  # TTS errors are typically transient

    def test_llm_provider_error_factory(self):
        """Should create LLM provider error with correct properties."""
        error = llm_provider_error("API quota exceeded")

        assert error.code == ErrorCode.LLM_PROVIDER_ERROR
        assert error.message == "LLM provider error"
        assert error.details == "API quota exceeded"
        assert error.retryable is True  # LLM errors are typically transient

    def test_configuration_error_factory(self):
        """Should create configuration error with correct properties."""
        error = configuration_error("GOOGLE_AI_API_KEY is required")

        assert error.code == ErrorCode.CONFIGURATION_ERROR
        assert error.message == "Configuration error"
        assert error.details == "GOOGLE_AI_API_KEY is required"
        assert error.retryable is False  # Configuration errors are not retryable


class TestErrorHandlingPatterns:
    """TDD tests for common error handling patterns."""

    def test_result_chaining_pattern(self):
        """Should support result chaining for error propagation."""

        def operation_that_fails() -> Result[str]:
            return Result.failure(text_extraction_error("PDF corrupted"))

        def operation_that_succeeds() -> Result[str]:
            return Result.success("success")

        # Test failure propagation
        failed_result = operation_that_fails()
        assert failed_result.is_failure
        assert failed_result.error is not None
        assert failed_result.error.code == ErrorCode.TEXT_EXTRACTION_FAILED

        # Test success
        success_result = operation_that_succeeds()
        assert success_result.is_success
        assert success_result.value == "success"

    def test_error_context_preservation(self):
        """Should preserve error context through function calls."""

        def low_level_operation() -> Result[str]:
            try:
                raise ConnectionError("Network unreachable")
            except Exception as e:
                return Result.from_exception(e, ErrorCode.TTS_ENGINE_ERROR, retryable=True)

        def high_level_operation() -> Result[str]:
            result = low_level_operation()
            if result.is_failure:
                # Error context is preserved
                return result
            return Result.success("processed")

        final_result = high_level_operation()
        assert final_result.is_failure
        assert final_result.error is not None
        assert final_result.error.code == ErrorCode.TTS_ENGINE_ERROR
        assert final_result.error.message == "Network unreachable"
        assert final_result.error.details == "ConnectionError"
        assert final_result.error.retryable is True

    def test_error_aggregation_pattern(self):
        """Should support collecting multiple errors."""
        errors = [
            text_extraction_error("Page 1 failed"),
            text_extraction_error("Page 3 failed"),
            audio_generation_error("TTS unavailable"),
        ]

        # Test that we can collect and analyze multiple errors
        assert len(errors) == 3

        extraction_errors = [e for e in errors if e.code == ErrorCode.TEXT_EXTRACTION_FAILED]
        audio_errors = [e for e in errors if e.code == ErrorCode.AUDIO_GENERATION_FAILED]

        assert len(extraction_errors) == 2
        assert len(audio_errors) == 1
        assert all(e.retryable for e in errors)  # All these errors are retryable

    def test_error_recovery_pattern(self):
        """Should support error recovery patterns."""

        def operation_with_fallback() -> Result[str]:
            # Primary operation fails
            primary_result: Result[str] = Result.failure(tts_engine_error("Gemini unavailable"))

            if primary_result.is_failure and primary_result.error is not None and primary_result.error.retryable:
                # Fallback operation succeeds
                return Result.success("fallback_result")

            return primary_result

        result = operation_with_fallback()
        assert result.is_success
        assert result.value == "fallback_result"


class TestErrorHandlingEdgeCases:
    """TDD tests for edge cases and unusual scenarios."""

    def test_error_with_very_long_messages(self):
        """Should handle very long error messages gracefully."""
        long_message = "x" * 10000  # 10k character message
        long_details = "y" * 5000  # 5k character details

        error = ApplicationError(code=ErrorCode.UNKNOWN_ERROR, message=long_message, details=long_details)

        assert len(error.message) == 10000
        assert error.details is not None
        assert len(error.details) == 5000
        assert str(error).startswith("unknown_error: " + "x" * 100)  # Should not crash

    def test_error_with_special_characters(self):
        """Should handle special characters in error messages."""
        special_chars = "!@#$%^&*()[]{}|;:,.<>?/~`"

        error = ApplicationError(
            code=ErrorCode.TEXT_EXTRACTION_FAILED,
            message=f"Error with special chars: {special_chars}",
            details=f"Details: {special_chars}",
        )

        assert special_chars in error.message
        assert error.details is not None
        assert special_chars in error.details
        assert special_chars in str(error)

    def test_result_with_complex_value_types(self):
        """Should handle complex value types in Result."""
        # Test with dictionary
        from typing import Any

        test_dict: dict[str, Any] = {"key": "value", "nested": {"inner": 42}}
        dict_result: Result[dict[str, Any]] = Result.success(test_dict)
        assert dict_result.is_success
        assert dict_result.value is not None
        assert dict_result.value["key"] == "value"
        assert dict_result.value["nested"]["inner"] == 42

        # Test with custom object
        class CustomObject:
            def __init__(self, name: str) -> None:
                self.name = name

        obj = CustomObject("test")
        obj_result = Result.success(obj)
        assert obj_result.is_success
        assert obj_result.value is not None
        assert obj_result.value.name == "test"

    def test_error_immutability_characteristics(self):
        """Should test error object characteristics."""
        error = ApplicationError(code=ErrorCode.CONFIGURATION_ERROR, message="Test error", retryable=False)

        # Document current behavior (dataclass is mutable by default)
        error.message = "Modified message"
        assert error.message == "Modified message"

        # This documents that errors are mutable (design decision)
        # If immutability is desired, this test should be updated

    def test_exception_to_result_with_none_exception(self):
        """Should handle edge cases in exception conversion."""

        # Test with exception that has None message
        class EmptyException(Exception):
            def __str__(self):
                return ""

        exception = EmptyException()
        result: Result[None] = Result.from_exception(exception)

        assert result.is_failure
        assert result.error is not None
        assert result.error.message == ""
        assert result.error.details == "EmptyException"
