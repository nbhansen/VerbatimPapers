"""DocumentEngine.extract_text: the PDF's own text, with OCR for pages that have too little."""

from unittest.mock import Mock

from domain.document.document_engine import DocumentEngine
from domain.errors import ErrorCode, Result, text_extraction_error
from domain.models import PdfPageText

PAGE_TEXT = "A page with plenty of real extracted text on it. " * 4  # Over the 100-char threshold


def _engine(pages: list[str], ocr: Result[str] | None = None, render: Result[bytes] | None = None) -> DocumentEngine:
    """Engine over a fake PDF whose pages have the given text layers."""
    extractor = Mock()
    extractor.extract_text_by_page.return_value = Result.success(
        [PdfPageText(page_index=i, text=text) for i, text in enumerate(pages)]
    )
    extractor.render_page_image.return_value = render or Result.success(b"png")
    ocr_provider = Mock()
    ocr_provider.perform_ocr.return_value = ocr or Result.success("Text recognized from the scanned page image.")
    return DocumentEngine(ocr_provider=ocr_provider, pdf_extractor=extractor)


class TestOcrFallback:
    """Pages whose text layer is missing or thin are read with OCR instead."""

    def test_page_with_enough_text_is_not_ocred(self) -> None:
        """A full text layer is used as is; OCR is slow and less accurate."""
        engine = _engine([PAGE_TEXT])

        result = engine.extract_text("doc.pdf")

        assert result.value == [PAGE_TEXT.strip()]
        engine.ocr_provider.perform_ocr.assert_not_called()  # type: ignore[attr-defined]

    def test_image_only_page_is_ocred(self) -> None:
        """A scanned page has no text layer, so its words come from OCR."""
        engine = _engine(["", PAGE_TEXT])

        result = engine.extract_text("doc.pdf")

        assert result.value == ["Text recognized from the scanned page image.", PAGE_TEXT.strip()]
        engine.pdf_extractor.render_page_image.assert_called_once_with("doc.pdf", 0, 300)  # type: ignore[attr-defined]

    def test_thin_text_layer_loses_to_longer_ocr(self) -> None:
        """A page with only a running header in its text layer is read with OCR."""
        result = _engine(["Journal 12"]).extract_text("doc.pdf")

        assert result.value == ["Text recognized from the scanned page image."]

    def test_thin_text_layer_kept_when_ocr_finds_less(self) -> None:
        """OCR only replaces the text layer when it recovers more."""
        result = _engine(["Short but real page text."], ocr=Result.success("Sh")).extract_text("doc.pdf")

        assert result.value == ["Short but real page text."]

    def test_ocr_failure_keeps_the_text_layer(self) -> None:
        """A failed OCR call leaves the page's own text in place rather than failing."""
        engine = _engine(["Short page.", PAGE_TEXT], ocr=Result.failure(text_extraction_error("tesseract died")))

        assert engine.extract_text("doc.pdf").value == ["Short page.", PAGE_TEXT.strip()]

    def test_render_failure_keeps_the_text_layer(self) -> None:
        """A page that can't be rendered is never handed to OCR."""
        engine = _engine(["Short page."], render=Result.failure(text_extraction_error("render failed")))

        assert engine.extract_text("doc.pdf").value == ["Short page."]
        engine.ocr_provider.perform_ocr.assert_not_called()  # type: ignore[attr-defined]


class TestEmptyPages:
    """Pages with no text at all are skipped; a document with none is an error."""

    def test_blank_page_is_skipped(self) -> None:
        """A blank separator page contributes nothing, not an empty string."""
        engine = _engine([PAGE_TEXT, "   ", PAGE_TEXT], ocr=Result.failure(text_extraction_error("blank")))

        assert engine.extract_text("doc.pdf").value == [PAGE_TEXT.strip(), PAGE_TEXT.strip()]

    def test_no_text_anywhere_is_an_extraction_failure(self) -> None:
        """An unreadable scan fails extraction instead of producing silent audio."""
        engine = _engine(["", " "], ocr=Result.failure(text_extraction_error("blank")))

        result = engine.extract_text("doc.pdf")

        assert result.is_failure and result.error is not None
        assert result.error.code == ErrorCode.TEXT_EXTRACTION_FAILED

    def test_requested_pages_reach_the_extractor(self) -> None:
        """Only the pages asked for are read."""
        engine = _engine([PAGE_TEXT])

        engine.extract_text("doc.pdf", [1, 2])

        engine.pdf_extractor.extract_text_by_page.assert_called_once_with("doc.pdf", [1, 2])  # type: ignore[attr-defined]
