"""What the user is told when a conversion fails."""

import pytest

from application.services.document_service import describe_failure
from domain.errors import (
    ApplicationError,
    ErrorCode,
    audio_generation_error,
    invalid_page_range_error,
    text_extraction_error,
)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            invalid_page_range_error("End page 9 is past the end of the document (3 pages)"),
            "Invalid page range for 'paper.pdf': End page 9 is past the end of the document (3 pages)",
        ),
        (
            text_extraction_error("The file could not be read as a PDF; it may be corrupt"),
            "Failed to extract text from PDF for 'paper.pdf': The file could not be read as a PDF; it may be corrupt",
        ),
        (
            audio_generation_error("Part 2 of 3 could not be synthesized: quota exceeded"),
            "Failed to generate audio for 'paper.pdf': Part 2 of 3 could not be synthesized: quota exceeded",
        ),
        (
            ApplicationError(code=ErrorCode.TEXT_CLEANING_FAILED, message="LLM unavailable"),
            "Failed to process and clean text for 'paper.pdf': LLM unavailable",
        ),
        (
            ApplicationError(
                code=ErrorCode.TEXT_EXTRACTION_FAILED, message="No /Root object!", details="PdfminerException"
            ),
            "Failed to extract text from PDF for 'paper.pdf': No /Root object!: PdfminerException",
        ),
    ],
)
def test_failure_names_the_stage_and_the_reason(error: ApplicationError, expected: str) -> None:
    """The stage comes from the error code, and the details say what to fix."""
    assert describe_failure(error, "paper.pdf") == expected
