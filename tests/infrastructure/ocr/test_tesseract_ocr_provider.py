# tests/infrastructure/ocr/test_tesseract_ocr_provider.py
"""Unit tests for TesseractOCRProvider: image bytes in, text out, failures as Result."""

import io
from unittest.mock import patch

from PIL import Image

from domain.errors import ErrorCode
from infrastructure.ocr.tesseract_ocr_provider import TesseractOCRProvider


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (20, 10), color=255).save(buffer, format="PNG")
    return buffer.getvalue()


class TestTesseractOCRProvider:
    """perform_ocr decodes the image in memory and hands it to Tesseract."""

    def test_default_language(self) -> None:
        """English unless configured otherwise."""
        assert TesseractOCRProvider().ocr_language == "eng"

    @patch("infrastructure.ocr.tesseract_ocr_provider.pytesseract.image_to_string")
    def test_recognizes_text_with_configured_language(self, mock_image_to_string) -> None:
        """The decoded image and the language reach Tesseract."""
        mock_image_to_string.return_value = "Texte français extrait"

        result = TesseractOCRProvider(language="fra").perform_ocr(_png())

        assert result.is_success
        assert result.value == "Texte français extrait"
        (image,) = mock_image_to_string.call_args.args
        assert isinstance(image, Image.Image)
        assert image.size == (20, 10)
        assert mock_image_to_string.call_args.kwargs == {"lang": "fra"}

    @patch("infrastructure.ocr.tesseract_ocr_provider.pytesseract.image_to_string")
    def test_blank_result_is_a_failure(self, mock_image_to_string) -> None:
        """No recognized text is reported, so the caller keeps the extracted text."""
        mock_image_to_string.return_value = "  \n\t "

        result = TesseractOCRProvider().perform_ocr(_png())

        assert result.is_failure and result.error is not None
        assert result.error.code == ErrorCode.TEXT_EXTRACTION_FAILED

    @patch("infrastructure.ocr.tesseract_ocr_provider.pytesseract.image_to_string")
    def test_tesseract_error_is_a_failure(self, mock_image_to_string) -> None:
        """A Tesseract crash becomes a failed Result, not an exception."""
        mock_image_to_string.side_effect = RuntimeError("tesseract is not installed")

        result = TesseractOCRProvider().perform_ocr(_png())

        assert result.is_failure and result.error is not None
        assert "tesseract is not installed" in (result.error.details or "")

    def test_undecodable_image_is_a_failure(self) -> None:
        """Bytes that aren't an image fail cleanly."""
        result = TesseractOCRProvider().perform_ocr(b"not an image")

        assert result.is_failure
