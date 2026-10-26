# infrastructure/pdf/pdfplumber_text_extractor.py - pdfplumber adapter for IPdfTextExtractor
"""PDF reading backed by pdfplumber: metadata, text and page images.

Implements the domain's IPdfTextExtractor interface so the domain layer stays
free of external library imports. All errors are returned as Result[T].
"""

import io
import logging

import pdfplumber

from domain.errors import ErrorCode, Result, text_extraction_error
from domain.interfaces import IPdfTextExtractor
from domain.models import PDFInfo, PdfPageText

logger = logging.getLogger(__name__)


class PdfPlumberTextExtractor(IPdfTextExtractor):
    """Reads PDFs using pdfplumber."""

    def get_pdf_info(self, pdf_path: str) -> Result[PDFInfo]:
        """Page count and metadata; an unreadable or page-less file is a failure, not 0 pages."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = pdf.metadata or {}
                info = PDFInfo(
                    total_pages=len(pdf.pages),
                    title=str(metadata.get("Title") or "Unknown"),
                    author=str(metadata.get("Author") or "Unknown"),
                )
        except Exception as e:
            logger.warning("Could not read PDF %s: %s", pdf_path, e)
            return Result.failure(text_extraction_error("The file could not be read as a PDF; it may be corrupt"))

        if info.total_pages == 0:
            return Result.failure(text_extraction_error("The PDF has no pages"))
        return Result.success(info)

    def extract_text_by_page(self, pdf_path: str, pages: list[int] | None = None) -> Result[list[PdfPageText]]:
        """Extract raw text for the requested 0-based pages (all pages if None).

        Out-of-range indices are skipped. A page whose extraction throws yields
        empty text instead of failing the whole document, so the caller can
        fall back to OCR for that page.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_indices = pages if pages is not None else range(len(pdf.pages))

                extracted: list[PdfPageText] = []
                for i in page_indices:
                    if not 0 <= i < len(pdf.pages):
                        continue
                    try:
                        text = pdf.pages[i].extract_text() or ""
                    except Exception:
                        logger.warning("Text extraction failed for page %d of %s", i + 1, pdf_path, exc_info=True)
                        text = ""
                    extracted.append(PdfPageText(page_index=i, text=text))

                return Result.success(extracted)

        except Exception as e:
            logger.exception("Error opening PDF %s: %s", pdf_path, e)
            return Result.from_exception(e, ErrorCode.TEXT_EXTRACTION_FAILED, retryable=True)

    def render_page_image(self, pdf_path: str, page_index: int, resolution: int = 300) -> Result[bytes]:
        """Render a single 0-based page as PNG bytes.

        Reopens the document per call; acceptable because rendering is only
        used for the occasional OCR-fallback page and OCR dominates the cost.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if not 0 <= page_index < len(pdf.pages):
                    return Result.failure(text_extraction_error(f"Page index {page_index} out of range for {pdf_path}"))

                img = pdf.pages[page_index].to_image(resolution=resolution).original
                with io.BytesIO() as buffer:
                    img.save(buffer, format="PNG")
                    return Result.success(buffer.getvalue())

        except Exception as e:
            logger.exception("Error rendering page %d of %s: %s", page_index, pdf_path, e)
            return Result.from_exception(e, ErrorCode.TEXT_EXTRACTION_FAILED, retryable=True)
