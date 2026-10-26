# tests/infrastructure/ocr/test_tesseract_ocr_provider.py
"""Comprehensive unit tests for TesseractOCRProvider implementation.

Tests image processing pipeline, PDF info extraction and page range validation.
"""

from unittest.mock import MagicMock, Mock, patch

from domain.errors import ErrorCode
from domain.models import PageRange
from infrastructure.ocr.tesseract_ocr_provider import TesseractOCRProvider


class TestTesseractOCRProviderInitialization:
    """Test TesseractOCRProvider initialization and configuration."""

    def test_init_with_default_settings(self):
        """Should initialize with default OCR settings."""
        provider = TesseractOCRProvider()

        assert provider.ocr_language == "eng"
        assert provider.poppler_path_custom is None

    def test_init_with_custom_tesseract_cmd(self):
        """Should set custom tesseract command path."""
        custom_cmd = "/usr/local/bin/tesseract"

        with patch("pytesseract.pytesseract") as mock_pytesseract:
            TesseractOCRProvider(tesseract_cmd=custom_cmd)

            assert mock_pytesseract.tesseract_cmd == custom_cmd

    def test_init_with_custom_poppler_path(self):
        """Should set custom poppler path."""
        custom_poppler = "/usr/local/bin"

        provider = TesseractOCRProvider(poppler_path_custom=custom_poppler)

        assert provider.poppler_path_custom == custom_poppler


class TestTesseractOCRProviderPerformOCR:
    """Test single image OCR functionality."""

    @patch("pytesseract.image_to_string")
    def test_perform_ocr_success(self, mock_image_to_string):
        """Should successfully perform OCR on image file."""
        mock_image_to_string.return_value = "Extracted text from image"
        provider = TesseractOCRProvider()

        result = provider.perform_ocr("test_image.png")

        assert result.is_success
        assert result.value == "Extracted text from image"
        mock_image_to_string.assert_called_once_with("test_image.png", lang="eng")

    @patch("pytesseract.image_to_string")
    def test_perform_ocr_with_custom_language(self, mock_image_to_string):
        """Should use custom language for OCR."""
        mock_image_to_string.return_value = "Texte français extrait"
        provider = TesseractOCRProvider(language="fra")
        result = provider.perform_ocr("french_image.png")

        assert result.is_success
        assert result.value == "Texte français extrait"
        mock_image_to_string.assert_called_once_with("french_image.png", lang="fra")

    @patch("pytesseract.image_to_string")
    def test_perform_ocr_empty_result(self, mock_image_to_string):
        """Should handle empty OCR result as failure."""
        mock_image_to_string.return_value = "   \n\t  "  # Whitespace only
        provider = TesseractOCRProvider()

        result = provider.perform_ocr("empty_image.png")

        assert result.is_failure
        assert result.error is not None
        assert result.error.code == ErrorCode.TEXT_EXTRACTION_FAILED
        assert result.error.details is not None
        assert "OCR process yielded no text" in result.error.details

    @patch("pytesseract.image_to_string")
    def test_perform_ocr_handles_pytesseract_exception(self, mock_image_to_string):
        """Should handle pytesseract exceptions gracefully."""
        mock_image_to_string.side_effect = Exception("Tesseract not found")
        provider = TesseractOCRProvider()

        result = provider.perform_ocr("bad_image.png")

        assert result.is_failure
        assert result.error is not None
        assert result.error.code == ErrorCode.TEXT_EXTRACTION_FAILED
        assert result.error.details is not None
        assert "OCR failed on bad_image.png: Tesseract not found" in result.error.details

    @patch("pytesseract.image_to_string")
    def test_perform_ocr_handles_file_not_found(self, mock_image_to_string):
        """Should handle file not found errors."""
        mock_image_to_string.side_effect = FileNotFoundError("Image file not found")
        provider = TesseractOCRProvider()

        result = provider.perform_ocr("nonexistent.png")

        assert result.is_failure
        assert result.error is not None
        assert result.error.details is not None
        assert "Image file not found" in result.error.details

    @patch("pytesseract.image_to_string")
    def test_perform_ocr_strips_whitespace_from_result(self, mock_image_to_string):
        """Should not strip whitespace from non-empty OCR results."""
        mock_image_to_string.return_value = "  Valid text with spaces  "
        provider = TesseractOCRProvider()

        result = provider.perform_ocr("spaced_text.png")

        assert result.is_success
        assert result.value == "  Valid text with spaces  "


class TestTesseractOCRProviderPDFInfo:
    """Test PDF information extraction."""

    @patch("pdfplumber.open")
    def test_get_pdf_info_success(self, mock_pdfplumber_open):
        """Should successfully extract PDF information."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock(), Mock(), Mock()]  # 3 pages
        mock_pdf.metadata = {"Title": "Test Document", "Author": "Test Author"}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        pdf_info = provider.get_pdf_info("test.pdf")

        assert pdf_info.total_pages == 3
        assert pdf_info.title == "Test Document"
        assert pdf_info.author == "Test Author"

    @patch("pdfplumber.open")
    def test_get_pdf_info_no_metadata(self, mock_pdfplumber_open):
        """Should handle PDF without metadata."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock()]  # 1 page
        mock_pdf.metadata = None
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        pdf_info = provider.get_pdf_info("no_metadata.pdf")

        assert pdf_info.total_pages == 1
        assert pdf_info.title == "Unknown"
        assert pdf_info.author == "Unknown"

    @patch("pdfplumber.open")
    def test_get_pdf_info_partial_metadata(self, mock_pdfplumber_open):
        """Should handle PDF with partial metadata."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock(), Mock()]  # 2 pages
        mock_pdf.metadata = {"Title": "Partial Info"}  # No Author
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        pdf_info = provider.get_pdf_info("partial.pdf")

        assert pdf_info.total_pages == 2
        assert pdf_info.title == "Partial Info"
        assert pdf_info.author == "Unknown"

    @patch("pdfplumber.open")
    def test_get_pdf_info_handles_exception(self, mock_pdfplumber_open):
        """Should handle exceptions during PDF info extraction."""
        mock_pdfplumber_open.side_effect = Exception("Corrupted PDF")

        provider = TesseractOCRProvider()
        pdf_info = provider.get_pdf_info("corrupted.pdf")

        assert pdf_info.total_pages == 0
        assert pdf_info.title == "Unknown"
        assert pdf_info.author == "Unknown"

    @patch("pdfplumber.open")
    def test_get_pdf_info_empty_pdf(self, mock_pdfplumber_open):
        """Should handle PDF with no pages."""
        mock_pdf = MagicMock()
        mock_pdf.pages = []  # No pages
        mock_pdf.metadata = {"Title": "Empty PDF"}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        pdf_info = provider.get_pdf_info("empty.pdf")

        assert pdf_info.total_pages == 0
        assert pdf_info.title == "Empty PDF"


class TestTesseractOCRProviderPageRangeValidation:
    """Test comprehensive page range validation logic."""

    @patch("pdfplumber.open")
    def test_validate_range_valid_full_document(self, mock_pdfplumber_open):
        """Should validate full document range."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock() for _ in range(10)]  # 10 pages
        mock_pdf.metadata = {}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        page_range = PageRange(start_page=None, end_page=None)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is True
        assert result["total_pages"] == 10
        assert result["actual_start"] == 1
        assert result["actual_end"] == 10
        assert result["pages_to_process"] == 10
        assert result["percentage_of_document"] == 100.0

    @patch("pdfplumber.open")
    def test_validate_range_valid_partial_range(self, mock_pdfplumber_open):
        """Should validate partial page range."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock() for _ in range(20)]  # 20 pages
        mock_pdf.metadata = {}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        page_range = PageRange(start_page=5, end_page=15)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is True
        assert result["total_pages"] == 20
        assert result["actual_start"] == 5
        assert result["actual_end"] == 15
        assert result["pages_to_process"] == 11  # 5 to 15 inclusive
        assert abs(result["percentage_of_document"] - 55.0) < 0.1

    @patch("pdfplumber.open")
    def test_validate_range_start_page_too_low(self, mock_pdfplumber_open):
        """Should reject start page less than 1."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock() for _ in range(10)]
        mock_pdf.metadata = {}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()

        # Create invalid range directly (assuming dataclass allows it)
        page_range = PageRange(start_page=0, end_page=5)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is False
        assert "Start page must be 1 or greater" in result["error"]

    @patch("pdfplumber.open")
    def test_validate_range_start_page_exceeds_total(self, mock_pdfplumber_open):
        """Should reject start page exceeding total pages."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock() for _ in range(5)]
        mock_pdf.metadata = {}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        page_range = PageRange(start_page=10, end_page=15)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is False
        assert "Start page 10 exceeds total pages (5)" in result["error"]
        assert result["total_pages"] == 5

    @patch("pdfplumber.open")
    def test_validate_range_end_page_too_low(self, mock_pdfplumber_open):
        """Should reject end page less than 1."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock() for _ in range(10)]
        mock_pdf.metadata = {}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()

        # Create invalid range directly
        page_range = PageRange(start_page=1, end_page=0)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is False
        assert "End page must be 1 or greater" in result["error"]

    @patch("pdfplumber.open")
    def test_validate_range_end_page_exceeds_total(self, mock_pdfplumber_open):
        """Should reject end page exceeding total pages."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock() for _ in range(8)]
        mock_pdf.metadata = {}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        page_range = PageRange(start_page=1, end_page=15)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is False
        assert "End page 15 exceeds total pages (8)" in result["error"]

    @patch("pdfplumber.open")
    def test_validate_range_start_greater_than_end(self, mock_pdfplumber_open):
        """Should reject start page greater than end page."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock() for _ in range(10)]
        mock_pdf.metadata = {}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()

        # Create invalid range directly
        page_range = PageRange(start_page=8, end_page=3)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is False
        assert "cannot be greater than" in result["error"]

    @patch("pdfplumber.open")
    def test_validate_range_zero_page_pdf(self, mock_pdfplumber_open):
        """Should handle PDF with zero pages."""
        mock_pdf = MagicMock()
        mock_pdf.pages = []  # No pages
        mock_pdf.metadata = {"Title": "Empty PDF"}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        page_range = PageRange(start_page=1, end_page=1)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is False
        assert "Could not determine PDF page count" in result["error"]
        assert result["total_pages"] == 0

    @patch.object(TesseractOCRProvider, "get_pdf_info")
    def test_validate_range_handles_exception(self, mock_get_pdf_info):
        """Should handle exceptions during validation."""
        mock_get_pdf_info.side_effect = Exception("PDF read error")

        provider = TesseractOCRProvider()
        page_range = PageRange(start_page=1, end_page=5)

        result = provider.validate_range("corrupted.pdf", page_range)

        assert result["valid"] is False
        assert "Page range validation failed: PDF read error" in result["error"]
        assert result["total_pages"] == 0

    @patch("pdfplumber.open")
    def test_validate_range_single_page_selection(self, mock_pdfplumber_open):
        """Should validate single page selection."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock() for _ in range(10)]
        mock_pdf.metadata = {}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        page_range = PageRange(start_page=5, end_page=5)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is True
        assert result["pages_to_process"] == 1
        assert result["percentage_of_document"] == 10.0

    @patch("pdfplumber.open")
    def test_validate_range_start_only_specified(self, mock_pdfplumber_open):
        """Should handle case where only start page is specified."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock() for _ in range(10)]
        mock_pdf.metadata = {}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        page_range = PageRange(start_page=3, end_page=None)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is True
        assert result["actual_start"] == 3
        assert result["actual_end"] == 10
        assert result["pages_to_process"] == 8

    @patch("pdfplumber.open")
    def test_validate_range_end_only_specified(self, mock_pdfplumber_open):
        """Should handle case where only end page is specified."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [Mock() for _ in range(10)]
        mock_pdf.metadata = {}
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        provider = TesseractOCRProvider()
        page_range = PageRange(start_page=None, end_page=7)

        result = provider.validate_range("test.pdf", page_range)

        assert result["valid"] is True
        assert result["actual_start"] == 1
        assert result["actual_end"] == 7
        assert result["pages_to_process"] == 7
