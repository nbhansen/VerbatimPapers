"""Tesseract OCR provider implementation for optical character recognition and PDF processing.

Combines direct text extraction with OCR fallback for reliable text extraction.
"""

import logging
from typing import Any

import pdfplumber
import pytesseract

from domain.errors import Result, text_extraction_error
from domain.interfaces import IOCRProvider
from domain.models import PageRange, PDFInfo

logger = logging.getLogger(__name__)


class TesseractOCRProvider(IOCRProvider):
    """OCR provider using Tesseract with PDF text extraction and validation capabilities."""

    def __init__(
        self,
        tesseract_cmd: str | None = None,
        poppler_path_custom: str | None = None,
        language: str = "eng",
    ) -> None:
        """Initialize the provider.

        Args:
            tesseract_cmd: Path to the tesseract binary (default: found on PATH)
            poppler_path_custom: Custom poppler binary directory
            language: Tesseract language code(s), e.g. "eng", "dan" or "eng+dan"
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.poppler_path_custom = poppler_path_custom
        self.ocr_language = language

        logger.debug("TesseractOCRProvider initialized: language=%s", self.ocr_language)

    def perform_ocr(self, image_path: str) -> Result[str]:
        """Perform OCR on a single image file."""
        logger.debug("Performing OCR on: %s", image_path)
        try:
            text = pytesseract.image_to_string(image_path, lang=self.ocr_language)
            if not text.strip():
                logger.warning("OCR yielded no text for: %s", image_path)
                return Result.failure(text_extraction_error("OCR process yielded no text"))
            logger.debug("OCR extracted %d chars from: %s", len(text), image_path)
            return Result.success(text)
        except Exception as e:
            logger.error("OCR failed on %s: %s", image_path, e)
            return Result.failure(text_extraction_error(f"OCR failed on {image_path}: {e!s}"))

    def get_pdf_info(self, pdf_path: str) -> PDFInfo:
        """Get basic PDF information."""
        logger.debug("Getting PDF info: %s", pdf_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                info = PDFInfo(
                    total_pages=len(pdf.pages),
                    title=pdf.metadata.get("Title", "Unknown") if pdf.metadata else "Unknown",
                    author=pdf.metadata.get("Author", "Unknown") if pdf.metadata else "Unknown",
                )
                logger.debug("PDF info: pages=%d, title=%s", info.total_pages, info.title)
                return info
        except Exception:
            logger.warning("Failed to get PDF info: %s", pdf_path, exc_info=True)
            return PDFInfo(total_pages=0, title="Unknown", author="Unknown")

    def validate_range(self, pdf_path: str, page_range: PageRange) -> dict[str, Any]:
        """Validate page range against PDF. Returns validation result."""
        logger.debug("Validating page range %s-%s for %s", page_range.start_page, page_range.end_page, pdf_path)
        try:
            pdf_info = self.get_pdf_info(pdf_path)
            total_pages = pdf_info.total_pages

            if total_pages == 0:
                return {"valid": False, "error": "Could not determine PDF page count", "total_pages": 0}

            # Validate start page
            if page_range.start_page is not None:
                if page_range.start_page < 1:
                    return self._error_result("Start page must be 1 or greater", total_pages)
                if page_range.start_page > total_pages:
                    return self._error_result(
                        f"Start page {page_range.start_page} exceeds total pages ({total_pages})", total_pages
                    )

            # Validate end page
            if page_range.end_page is not None:
                if page_range.end_page < 1:
                    return self._error_result("End page must be 1 or greater", total_pages)
                if page_range.end_page > total_pages:
                    return self._error_result(
                        f"End page {page_range.end_page} exceeds total pages ({total_pages})", total_pages
                    )

            # Validate range consistency
            if (
                page_range.start_page is not None
                and page_range.end_page is not None
                and page_range.start_page > page_range.end_page
            ):
                return self._error_result(
                    f"Start page ({page_range.start_page}) cannot be greater than end page ({page_range.end_page})",
                    total_pages,
                )

            # All validations passed
            actual_start = page_range.start_page if page_range.start_page is not None else 1
            actual_end = page_range.end_page if page_range.end_page is not None else total_pages

            return {
                "valid": True,
                "total_pages": total_pages,
                "actual_start": actual_start,
                "actual_end": actual_end,
                "pages_to_process": actual_end - actual_start + 1,
                "percentage_of_document": ((actual_end - actual_start + 1) / total_pages) * 100,
            }

        except Exception as e:
            return {"valid": False, "error": f"Page range validation failed: {e!s}", "total_pages": 0}

    def _error_result(self, error: str, total_pages: int) -> dict[str, Any]:
        return {"valid": False, "error": error, "total_pages": total_pages}
