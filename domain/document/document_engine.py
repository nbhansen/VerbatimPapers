# domain/document/document_engine.py - Document Processing Engine with Result[T] pattern
"""Document engine using Result[T] pattern for type-safe error handling.

No exceptions thrown - all errors returned as Result[T].
"""

from contextlib import suppress
import logging
from typing import Any

from ..errors import (
    ErrorCode,
    Result,
    audio_generation_error,
    text_extraction_error,
)
from ..interfaces import (
    IAudioEngine,
    IDocumentEngine,
    IFileManager,
    IOCRProvider,
    IPdfTextExtractor,
    ITextPipeline,
)
from ..models import PageRange, PDFInfo, ProcessingRequest, TimedAudioResult

logger = logging.getLogger(__name__)


class DocumentEngine(IDocumentEngine):
    """Document engine using Result[T] pattern for all operations.

    Pure functions with no exceptions - all errors returned as Result[T].
    """

    def __init__(
        self,
        ocr_provider: IOCRProvider,
        file_manager: IFileManager,
        pdf_extractor: IPdfTextExtractor,
        min_text_threshold: int = 100,
        audio_target_chunk_size: int = 4000,
        ocr_resolution: int = 300,
    ):
        self.ocr_provider = ocr_provider
        self.file_manager = file_manager
        self.pdf_extractor = pdf_extractor
        self.min_text_threshold = min_text_threshold
        self.audio_target_chunk_size = audio_target_chunk_size
        self.ocr_resolution = ocr_resolution  # DPI at which pages are rendered for OCR
        logger.debug("DocumentEngine initialized with Result[T] pattern")

    def get_pdf_info(self, pdf_path: str) -> Result[PDFInfo]:
        """Get PDF information - delegates to OCR provider."""
        try:
            info = self.ocr_provider.get_pdf_info(pdf_path)
            return Result.success(info)
        except Exception as e:
            return Result.from_exception(e, ErrorCode.TEXT_EXTRACTION_FAILED, retryable=True)

    def validate_page_range(self, pdf_path: str, page_range: PageRange) -> Result[dict[str, Any]]:
        """Validate page range - delegates to OCR provider."""
        try:
            validation = self.ocr_provider.validate_range(pdf_path, page_range)
            return Result.success(validation)
        except Exception as e:
            return Result.from_exception(e, ErrorCode.INVALID_PAGE_RANGE, retryable=False)

    def extract_text(self, pdf_path: str, pages: list[int] | None = None) -> Result[list[str]]:
        """Extract text from PDF with intelligent OCR fallback.

        Returns Result with extracted text or error.
        """
        pages_result = self.pdf_extractor.extract_text_by_page(pdf_path, pages)
        if pages_result.is_failure:
            return Result.failure(pages_result.error)  # type: ignore[arg-type]

        extracted_text = []
        for page in pages_result.value or []:
            text = page.text

            # If direct extraction is poor, try OCR fallback
            if not text or len(text.strip()) < self.min_text_threshold:
                logger.debug("Page %d has low text quality, using OCR", page.page_index + 1)
                ocr_text = self._ocr_page(pdf_path, page.page_index)
                # Use whichever text is longer
                if len(ocr_text) > len(text):
                    text = ocr_text

            if text.strip():
                extracted_text.append(text.strip())
            # Continue on empty pages - partial extraction is better than none

        if not extracted_text:
            return Result.failure(text_extraction_error("No text could be extracted from the PDF"))

        return Result.success(extracted_text)

    def process_document(
        self,
        request: ProcessingRequest,
        audio_engine: IAudioEngine,
        text_pipeline: ITextPipeline,
        llm_chunk_size: int = 50000,
    ) -> Result[TimedAudioResult]:
        """Complete document processing workflow using Result[T] pattern.

        Returns Result with audio result or error - no exceptions thrown.
        """
        # Extract and validate text from PDF
        text_result = self._extract_initial_text(request)
        if text_result.is_failure:
            return Result.failure(text_result.error)  # type: ignore[arg-type]
        text_chunks = text_result.value

        if text_chunks is None:
            return Result.failure(text_extraction_error("Text extraction returned no chunks"))

        # Process text through LLM cleaning and optimization pipeline
        processed_result = self._process_text_pipeline(text_chunks, text_pipeline, llm_chunk_size)
        if processed_result.is_failure:
            return Result.failure(processed_result.error)  # type: ignore[arg-type]
        processed_chunks = processed_result.value

        if processed_chunks is None:
            return Result.failure(audio_generation_error("Text processing returned no chunks"))

        audio_result = self._generate_audio_output(processed_chunks, request.output_name, audio_engine)
        if audio_result.is_failure:
            return Result.failure(audio_result.error)  # type: ignore[arg-type]

        return audio_result

    def _extract_initial_text(self, request: ProcessingRequest) -> Result[list[str]]:
        """Extract and validate text from PDF with page range conversion."""
        logger.info("Starting processing for %s", request.pdf_path)

        # Convert page range to page list
        pages_result = self._convert_page_range_to_list(request.pdf_path, request.page_range)
        if pages_result.is_failure:
            return Result.failure(pages_result.error)  # type: ignore[arg-type]
        pages_list = pages_result.value

        # Extract text from PDF
        text_result = self.extract_text(request.pdf_path, pages_list)
        if text_result.is_failure:
            return text_result

        text_chunks = text_result.value
        if not text_chunks:
            logger.warning("No text extracted from PDF")
            return Result.failure(text_extraction_error("No text could be extracted from the PDF"))

        logger.info("Extracted %d text chunks", len(text_chunks))
        return Result.success(text_chunks)

    def _process_text_pipeline(
        self, text_chunks: list[str], text_pipeline: ITextPipeline, llm_chunk_size: int
    ) -> Result[list[str]]:
        """Process text through LLM cleaning and optimization pipeline."""
        logger.debug("Using optimized chunking strategy (LLM chunk size: %d)", llm_chunk_size)

        try:
            # Step 1: Combine chunks for LLM processing
            combined_chunks = self._combine_chunks_for_llm(text_chunks, llm_chunk_size)
            logger.debug("Combined %d original chunks into %d LLM chunks", len(text_chunks), len(combined_chunks))

            # Step 2: Process through LLM cleaning
            cleaned_chunks = []
            for i, combined_chunk in enumerate(combined_chunks, 1):
                logger.debug("Processing LLM chunk %d/%d (%d chars)", i, len(combined_chunks), len(combined_chunk))
                logger.debug("Combined text preview: '%s...'", combined_chunk[:100])

                logger.debug("Calling text_pipeline.clean_text()...")
                clean_result = text_pipeline.clean_text(combined_chunk)
                if clean_result.is_failure:
                    return Result.failure(clean_result.error)  # type: ignore[arg-type]

                cleaned = clean_result.value
                if cleaned is not None:
                    logger.debug("Cleaned text (%d chars): '%s...'", len(cleaned), cleaned[:100])
                    cleaned_chunks.append(cleaned)
                else:
                    logger.debug("Cleaning returned None, skipping chunk")

            # Step 3: Re-combine all cleaned text
            all_cleaned_text = " ".join(cleaned_chunks)
            logger.debug("Combined all cleaned text: %d chars total", len(all_cleaned_text))

            # An image-only PDF extracts as whitespace and cleans down to nothing. Without
            # this, _split_for_tts("") returns [] and the caller only rejects None, so the
            # run reports success having produced no audio at all.
            if not all_cleaned_text.strip():
                return Result.failure(text_extraction_error("No usable text remained after cleaning"))

            # Step 4: Split cleaned text into optimal chunks for TTS
            processed_chunks = self._split_for_tts(all_cleaned_text, self.audio_target_chunk_size)
            logger.debug("Split cleaned text into %d TTS-optimized chunks", len(processed_chunks))

            logger.info(
                "Processed through optimized pipeline: %d -> %d -> %d chunks",
                len(text_chunks),
                len(combined_chunks),
                len(processed_chunks),
            )

            return Result.success(processed_chunks)

        except Exception as e:
            return Result.from_exception(e, ErrorCode.TEXT_CLEANING_FAILED, retryable=True)

    def _generate_audio_output(
        self, processed_chunks: list[str], output_name: str, audio_engine: IAudioEngine
    ) -> Result[TimedAudioResult]:
        """Generate the audio and its timeline."""
        try:
            audio_result = audio_engine.generate(processed_chunks, output_name)

            logger.debug("audio_result=%s", audio_result)

            if audio_result.is_failure:
                return audio_result

            timed_result = audio_result.value
            if timed_result:
                logger.debug("timed_result.audio_files=%s", timed_result.audio_files)

            if not timed_result or not timed_result.audio_files:
                return Result.failure(audio_generation_error("Audio generation failed to produce files"))

            return Result.success(timed_result)

        except Exception as e:
            return Result.from_exception(e, ErrorCode.AUDIO_GENERATION_FAILED, retryable=True)

    def _convert_page_range_to_list(self, pdf_path: str, page_range: PageRange) -> Result[list[int] | None]:
        """Convert PageRange to 0-based page list."""
        if page_range.is_full_document():
            return Result.success(None)

        start = page_range.start_page or 1
        end = page_range.end_page

        if end is None:
            # Get total pages to determine end
            pdf_info_result = self.get_pdf_info(pdf_path)
            if pdf_info_result.is_failure:
                return Result.failure(pdf_info_result.error)  # type: ignore[arg-type]
            pdf_info = pdf_info_result.value
            if pdf_info is None:
                return Result.failure(text_extraction_error("Could not get PDF info"))
            end = pdf_info.total_pages

        return Result.success(list(range(start - 1, end)))  # Convert to 0-based indexing

    def _ocr_page(self, pdf_path: str, page_index: int) -> str:
        """Perform OCR on a single PDF page, returning empty text on any failure."""
        temp_image_path = None

        try:
            image_result = self.pdf_extractor.render_page_image(pdf_path, page_index, self.ocr_resolution)
            if image_result.is_failure or image_result.value is None:
                logger.warning("Page image rendering failed: %s", image_result.error)
                return ""

            temp_image_path = self.file_manager.save_temp_file(image_result.value, suffix=".png")

            # Perform OCR using the provider
            ocr_result = self.ocr_provider.perform_ocr(temp_image_path)
            if ocr_result.is_success:
                return ocr_result.value or ""
            else:
                logger.warning("OCR provider failed: %s", ocr_result.error)
                return ""

        except Exception as e:
            logger.exception("OCR failed for page: %s", e)
            return ""

        finally:
            # Clean up temporary file
            if temp_image_path:
                # Ignore file cleanup errors - temporary files may already be removed
                with suppress(OSError, FileNotFoundError):
                    self.file_manager.delete_file(temp_image_path)

    def _combine_chunks_for_llm(self, text_chunks: list[str], llm_chunk_size: int) -> list[str]:
        """Combine small PDF chunks into larger chunks optimal for LLM processing.

        Pure function - no exceptions thrown.
        """
        if not text_chunks:
            return []

        combined_chunks = []
        current_chunk = ""

        for chunk in text_chunks:
            # Check if adding this chunk would exceed the target size
            if current_chunk and len(current_chunk) + len(chunk) + 1 > llm_chunk_size:
                # Current chunk is full, start a new one
                combined_chunks.append(current_chunk.strip())
                current_chunk = chunk
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + chunk
                else:
                    current_chunk = chunk

        # Add the final chunk if it has content
        if current_chunk.strip():
            combined_chunks.append(current_chunk.strip())

        return combined_chunks

    def _split_for_tts(self, text: str, target_chunk_size: int = 4000) -> list[str]:
        """Split enhanced text into chunks optimal for TTS processing.

        Pure function - no exceptions thrown.
        """
        if not text:
            return []

        # Simple sentence-based splitting that handles SSML markup
        import re

        # Split on sentence boundaries, being careful with SSML tags
        sentences = re.split(r"([.!?]+(?:\s*(?:<[^>]*>)?\s*))", text)

        # Rejoin sentences with their delimiters
        sentence_list = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            sentence_list.append(sentence.strip())

        # Combine sentences into chunks of target size
        chunks = []
        current_chunk = ""

        for sentence in sentence_list:
            if not sentence:
                continue

            # Check if adding this sentence would exceed target size
            if current_chunk and len(current_chunk) + len(sentence) + 1 > target_chunk_size:
                # Current chunk is full, start a new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add the final chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Fallback: if no chunks created, split by character count
        if not chunks and text.strip():
            logger.debug("Sentence splitting failed, using character-based fallback")
            for i in range(0, len(text), target_chunk_size):
                chunk = text[i : i + target_chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())

        return chunks
