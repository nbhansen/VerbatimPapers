"""Tesseract OCR provider: recognizes text in page images."""

import io
import logging

from PIL import Image
import pytesseract

from domain.errors import Result, text_extraction_error
from domain.interfaces import IOCRProvider

logger = logging.getLogger(__name__)


class TesseractOCRProvider(IOCRProvider):
    """OCR provider using the Tesseract binary via pytesseract."""

    def __init__(self, language: str = "eng") -> None:
        """Initialize the provider.

        Args:
            language: Tesseract language code(s), e.g. "eng", "dan" or "eng+dan"
        """
        self.ocr_language = language
        logger.debug("TesseractOCRProvider initialized: language=%s", self.ocr_language)

    def perform_ocr(self, image: bytes) -> Result[str]:
        """Recognize the text in an encoded image, in memory."""
        try:
            with Image.open(io.BytesIO(image)) as img:
                text = pytesseract.image_to_string(img, lang=self.ocr_language)
        except Exception as e:
            logger.error("OCR failed: %s", e)
            return Result.failure(text_extraction_error(f"OCR failed: {e!s}"))

        if not text.strip():
            logger.warning("OCR yielded no text")
            return Result.failure(text_extraction_error("OCR process yielded no text"))
        logger.debug("OCR extracted %d chars", len(text))
        return Result.success(text)
