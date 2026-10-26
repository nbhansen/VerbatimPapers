"""PageRange validation: out-of-range pages are reported, never silently clipped (#77)."""

from unittest.mock import Mock

import pytest

from domain.document.document_engine import DocumentEngine
from domain.errors import ErrorCode, Result, text_extraction_error
from domain.models import PageRange, PDFInfo, ProcessingRequest


class TestValidateAgainstDocument:
    """PageRange resolves to 0-based pages or explains what doesn't fit."""

    @pytest.mark.parametrize(
        ("page_range", "expected"),
        [
            (PageRange(), [0, 1, 2, 3, 4]),
            (PageRange(start_page=2, end_page=4), [1, 2, 3]),
            (PageRange(start_page=4), [3, 4]),
            (PageRange(end_page=2), [0, 1]),
            (PageRange(start_page=5, end_page=5), [4]),
        ],
    )
    def test_resolves_pages(self, page_range: PageRange, expected: list[int]) -> None:
        """Open ends default to the document's first and last page."""
        result = page_range.validate_against_document(total_pages=5)

        assert result.is_success
        assert result.value == expected

    @pytest.mark.parametrize(
        ("page_range", "message"),
        [
            (PageRange(start_page=2, end_page=9), "End page 9 is past the end of the document (5 pages)"),
            (PageRange(start_page=7), "Start page 7 is past the end of the document (5 pages)"),
            (PageRange(start_page=4, end_page=2), "Start page 4 is after end page 2"),
            (PageRange(start_page=0, end_page=2), "Page numbers start at 1"),
        ],
    )
    def test_reports_ranges_that_dont_fit(self, page_range: PageRange, message: str) -> None:
        """The error says exactly what is wrong, so the user can fix it."""
        result = page_range.validate_against_document(total_pages=5)

        assert result.is_failure and result.error is not None
        assert result.error.code == ErrorCode.INVALID_PAGE_RANGE
        assert result.error.details == message


class TestDocumentEngineUsesRealPageCount:
    """DocumentEngine checks the requested range against the PDF before extracting."""

    @staticmethod
    def _engine(info: Result[PDFInfo]) -> tuple[DocumentEngine, Mock]:
        extractor = Mock()
        extractor.get_pdf_info.return_value = info
        return DocumentEngine(ocr_provider=Mock(), pdf_extractor=extractor), extractor

    def test_end_page_beyond_document_fails_without_extracting(self) -> None:
        """Asking for pages 2-9 of a 5-page PDF is an error, not pages 2-5."""
        engine, extractor = self._engine(Result.success(PDFInfo(total_pages=5, title="t", author="a")))
        request = ProcessingRequest(pdf_path="doc.pdf", output_name="doc", page_range=PageRange(2, 9))

        result = engine.process_document(request, audio_engine=Mock(), text_pipeline=Mock())

        assert result.is_failure and result.error is not None
        assert result.error.code == ErrorCode.INVALID_PAGE_RANGE
        extractor.extract_text_by_page.assert_not_called()

    def test_unreadable_pdf_fails_the_page_lookup(self) -> None:
        """A corrupt PDF stops processing with the extractor's error."""
        engine, _ = self._engine(Result.failure(text_extraction_error("The PDF has no pages")))
        request = ProcessingRequest(pdf_path="doc.pdf", output_name="doc", page_range=PageRange(1, 2))

        result = engine.process_document(request, audio_engine=Mock(), text_pipeline=Mock())

        assert result.is_failure and result.error is not None
        assert result.error.details == "The PDF has no pages"
