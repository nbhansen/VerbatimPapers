"""Tests for the pdfplumber-backed IPdfTextExtractor adapter."""

from unittest.mock import MagicMock, Mock, patch

from domain.models import PDFInfo, PdfPageText
from infrastructure.pdf.pdfplumber_text_extractor import PdfPlumberTextExtractor


def make_mock_pdf(page_texts: list[str | None]) -> MagicMock:
    """Build a mock pdfplumber PDF whose pages return the given texts."""
    mock_pdf = MagicMock()
    pages = []
    for text in page_texts:
        page = Mock()
        page.extract_text.return_value = text
        pages.append(page)
    mock_pdf.pages = pages
    return mock_pdf


class TestExtractTextByPage:
    """Tests for extract_text_by_page."""

    @patch("pdfplumber.open")
    def test_extracts_all_pages_by_default(self, mock_open):
        """Should extract every page when no page list is given."""
        mock_open.return_value.__enter__.return_value = make_mock_pdf(["page one", "page two"])

        result = PdfPlumberTextExtractor().extract_text_by_page("test.pdf")

        assert result.is_success
        assert result.value == [
            PdfPageText(page_index=0, text="page one"),
            PdfPageText(page_index=1, text="page two"),
        ]

    @patch("pdfplumber.open")
    def test_extracts_requested_pages_only(self, mock_open):
        """Should extract only the requested page indices."""
        mock_open.return_value.__enter__.return_value = make_mock_pdf(["a", "b", "c"])

        result = PdfPlumberTextExtractor().extract_text_by_page("test.pdf", pages=[0, 2])

        assert result.is_success
        assert result.value == [
            PdfPageText(page_index=0, text="a"),
            PdfPageText(page_index=2, text="c"),
        ]

    @patch("pdfplumber.open")
    def test_skips_out_of_range_pages(self, mock_open):
        """Should silently skip out-of-range indices."""
        mock_open.return_value.__enter__.return_value = make_mock_pdf(["only page"])

        result = PdfPlumberTextExtractor().extract_text_by_page("test.pdf", pages=[0, 5, -1])

        assert result.is_success
        assert result.value == [PdfPageText(page_index=0, text="only page")]

    @patch("pdfplumber.open")
    def test_empty_page_list_extracts_nothing(self, mock_open):
        """An explicit empty page list must extract no pages, not the whole document."""
        mock_open.return_value.__enter__.return_value = make_mock_pdf(["a", "b"])

        result = PdfPlumberTextExtractor().extract_text_by_page("test.pdf", pages=[])

        assert result.is_success
        assert result.value == []

    @patch("pdfplumber.open")
    def test_none_text_becomes_empty_string(self, mock_open):
        """Should normalize None from extract_text() to an empty string."""
        mock_open.return_value.__enter__.return_value = make_mock_pdf([None])

        result = PdfPlumberTextExtractor().extract_text_by_page("test.pdf")

        assert result.is_success
        assert result.value == [PdfPageText(page_index=0, text="")]

    @patch("pdfplumber.open")
    def test_page_extraction_failure_yields_empty_text(self, mock_open):
        """A failing page should yield empty text, not fail the document."""
        mock_pdf = make_mock_pdf(["good page"])
        broken_page = Mock()
        broken_page.extract_text.side_effect = Exception("corrupt page")
        mock_pdf.pages = [broken_page, *mock_pdf.pages]
        mock_open.return_value.__enter__.return_value = mock_pdf

        result = PdfPlumberTextExtractor().extract_text_by_page("test.pdf")

        assert result.is_success
        assert result.value == [
            PdfPageText(page_index=0, text=""),
            PdfPageText(page_index=1, text="good page"),
        ]

    @patch("pdfplumber.open")
    def test_open_failure_returns_error(self, mock_open):
        """A document that cannot be opened should return a failure Result."""
        mock_open.side_effect = Exception("not a PDF")

        result = PdfPlumberTextExtractor().extract_text_by_page("bad.pdf")

        assert result.is_failure


class TestRenderPageImage:
    """Tests for render_page_image."""

    @patch("pdfplumber.open")
    def test_renders_page_as_png_bytes(self, mock_open):
        """Should render the page at 300dpi and return PNG bytes."""
        mock_pdf = make_mock_pdf(["page"])
        mock_img = Mock()
        mock_img.save.side_effect = lambda buffer, format: buffer.write(b"png-bytes")
        mock_pdf.pages[0].to_image.return_value.original = mock_img
        mock_open.return_value.__enter__.return_value = mock_pdf

        result = PdfPlumberTextExtractor().render_page_image("test.pdf", 0)

        assert result.is_success
        assert result.value == b"png-bytes"
        mock_pdf.pages[0].to_image.assert_called_once_with(resolution=300)

    @patch("pdfplumber.open")
    def test_out_of_range_page_returns_error(self, mock_open):
        """An out-of-range page index should return a failure Result."""
        mock_open.return_value.__enter__.return_value = make_mock_pdf(["page"])

        result = PdfPlumberTextExtractor().render_page_image("test.pdf", 3)

        assert result.is_failure

    @patch("pdfplumber.open")
    def test_render_failure_returns_error(self, mock_open):
        """A rendering exception should return a failure Result."""
        mock_pdf = make_mock_pdf(["page"])
        mock_pdf.pages[0].to_image.side_effect = Exception("render boom")
        mock_open.return_value.__enter__.return_value = mock_pdf

        result = PdfPlumberTextExtractor().render_page_image("test.pdf", 0)

        assert result.is_failure


class TestGetPdfInfo:
    """Page count and metadata, with unreadable files reported as failures."""

    @patch("infrastructure.pdf.pdfplumber_text_extractor.pdfplumber.open")
    def test_reads_page_count_and_metadata(self, mock_open):
        """Title and author come from the PDF metadata."""
        pdf = make_mock_pdf(["a", "b", "c"])
        pdf.metadata = {"Title": "A Paper", "Author": "Someone"}
        mock_open.return_value.__enter__.return_value = pdf

        result = PdfPlumberTextExtractor().get_pdf_info("doc.pdf")

        assert result.is_success
        assert result.value == PDFInfo(total_pages=3, title="A Paper", author="Someone")

    @patch("infrastructure.pdf.pdfplumber_text_extractor.pdfplumber.open")
    def test_missing_metadata_defaults_to_unknown(self, mock_open):
        """A PDF without metadata still reports its pages."""
        pdf = make_mock_pdf(["a"])
        pdf.metadata = None
        mock_open.return_value.__enter__.return_value = pdf

        result = PdfPlumberTextExtractor().get_pdf_info("doc.pdf")

        assert result.value == PDFInfo(total_pages=1, title="Unknown", author="Unknown")

    def test_corrupt_file_is_a_failure_not_zero_pages(self, tmp_path):
        """A file that isn't a PDF fails, rather than succeeding with 0 pages."""
        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"%PDF-1.4 this is not really a pdf")

        result = PdfPlumberTextExtractor().get_pdf_info(str(corrupt))

        assert result.is_failure and result.error is not None
        assert "could not be read as a PDF" in (result.error.details or "")

    @patch("infrastructure.pdf.pdfplumber_text_extractor.pdfplumber.open")
    def test_zero_pages_is_a_failure(self, mock_open):
        """A page-less PDF has nothing to convert."""
        mock_open.return_value.__enter__.return_value = make_mock_pdf([])

        result = PdfPlumberTextExtractor().get_pdf_info("empty.pdf")

        assert result.is_failure
