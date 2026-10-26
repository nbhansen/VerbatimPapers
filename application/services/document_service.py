"""Application service for document processing orchestration.

This service decouples the web layer from the core domain logic by providing
a unified interface for processing documents into audio.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.config.system_config import SystemConfig
    from domain.models import TimingMetadata

from application.services.error_formatting import clean_text_for_display, get_processing_stage_error
from application.services.progress_store import ThreadSafeProgressStore
from domain.errors import ApplicationError, ErrorCode
from domain.interfaces import IAudioEngine, IDocumentEngine, ITextPipeline
from domain.models import ProcessingRequest
from utils import parse_page_range_from_form, parse_plain_english_from_form

logger = logging.getLogger(__name__)


def output_base_name(original_filename: str, operation_id: str) -> str:
    """Base name for a conversion's output files: the upload's stem plus a short operation id."""
    return f"{Path(original_filename).stem}_{operation_id[:8]}"


# Which stage a failed conversion is reported under, by error code
_FAILURE_STAGES = {
    ErrorCode.INVALID_PAGE_RANGE: "page_range",
    ErrorCode.TEXT_EXTRACTION_FAILED: "text_extraction",
    ErrorCode.TEXT_CLEANING_FAILED: "text_processing",
}


# The fixed messages of domain.errors' helpers, which only restate the stage
_GENERIC_MESSAGES = frozenset(
    {
        "Failed to extract text from PDF",
        "Invalid page range",
        "Audio generation failed",
        "TTS engine error",
        "LLM provider error",
        "Configuration error",
    }
)


def describe_failure(error: ApplicationError | None, original_filename: str) -> str:
    """User-facing message for a failed conversion: the stage that failed and why."""
    if error is None:
        return get_processing_stage_error("processing", Exception("unknown error"), original_filename)
    stage = _FAILURE_STAGES.get(error.code, "audio_generation")
    if error.details and error.message in _GENERIC_MESSAGES:
        reason = error.details  # The stage already says what failed; the details say why
    elif error.details:
        reason = f"{error.message}: {error.details}"  # from_exception: exception text, then its type
    else:
        reason = error.message
    return get_processing_stage_error(stage, Exception(reason), original_filename)


def timing_json_filename(base_filename: str) -> str:
    """Name of the read-along timing file saved for a conversion's base filename."""
    return f"{base_filename}_timing.json"


class DocumentProcessingService:
    """Service for orchestrating document-to-audio conversion pipeline."""

    def __init__(
        self,
        document_engine: IDocumentEngine,
        audio_engine: IAudioEngine,
        text_pipeline: ITextPipeline,
        plain_english_text_pipeline: ITextPipeline,
        progress_store: ThreadSafeProgressStore,
        config: "SystemConfig",
    ):
        self.document_engine = document_engine
        self.audio_engine = audio_engine
        self.text_pipeline = text_pipeline
        self.plain_english_text_pipeline = plain_english_text_pipeline
        self.progress_store = progress_store
        self.config = config

    def process_document_background(
        self,
        operation_id: str,
        request_form: dict[str, str],
        saved_file_path: str,
        original_filename: str,
    ) -> None:
        """Process a document in the background with progress reporting."""
        try:
            logger.info("Starting document processing for operation %s, file: %s", operation_id, original_filename)
            self.progress_store.update(operation_id, "starting", 5, "Initializing document processing...")

            if self.progress_store.is_cancelled(operation_id):
                return

            # 1. Validate and prepare file info (10%)
            self.progress_store.update(operation_id, "validating", 10, "Validating uploaded file...")

            # The operation id keeps outputs of same-named uploads from overwriting each other
            base_filename = output_base_name(original_filename, operation_id)
            page_range = parse_page_range_from_form(request_form)
            enable_plain_english = parse_plain_english_from_form(request_form)

            if self.progress_store.is_cancelled(operation_id):
                return

            # 2. Execute processing (20-95%)
            text_pipeline = self.text_pipeline
            if enable_plain_english:
                text_pipeline = self.plain_english_text_pipeline
                logger.info("Using TextPipeline with plain English conversion enabled")

            self.progress_store.update(operation_id, "processing", 20, "Starting document processing...")

            request_obj = ProcessingRequest(pdf_path=saved_file_path, output_name=base_filename, page_range=page_range)
            logger.debug(
                "Processing request: pdf=%s, output=%s, pages=%s-%s",
                request_obj.pdf_path,
                request_obj.output_name,
                page_range.start_page,
                page_range.end_page,
            )

            # Core processing
            processing_result = self.document_engine.process_document(
                request_obj, self.audio_engine, text_pipeline, self.config.text_processing.llm_chunk_size
            )

            if self.progress_store.is_cancelled(operation_id):
                return

            logger.debug("Processing result: success=%s", processing_result.is_success)
            if processing_result.is_failure:
                enhanced_error = describe_failure(processing_result.error, original_filename)
                self.progress_store.update(
                    operation_id,
                    "error",
                    0,
                    "Processing failed",
                    is_error=True,
                    error_message=enhanced_error,
                )
                return

            # 3. Finalize result (100%)
            audio_result = processing_result.value
            if audio_result is None:
                self.progress_store.update(
                    operation_id,
                    "error",
                    0,
                    "Processing failed",
                    is_error=True,
                    error_message="Processing succeeded but returned no audio result",
                )
                return

            # Every conversion gets a read-along timeline
            if audio_result.timing_data:
                self._save_timing_data(base_filename, audio_result.timing_data)

            self.progress_store.update(
                operation_id,
                "complete",
                100,
                "Processing complete!",
                is_complete=True,
                result_data={
                    "base_filename": base_filename,
                    "original_filename": original_filename,
                    "audio_result": audio_result,
                    "page_range": page_range,
                },
            )

        except Exception as e:
            logger.exception("Unexpected error in background processing: %s", e)
            enhanced_error = get_processing_stage_error("processing", e, original_filename)
            self.progress_store.update(
                operation_id,
                "error",
                0,
                "Processing failed unexpectedly",
                is_error=True,
                error_message=enhanced_error,
            )
        finally:
            # Always clean up the uploaded file when processing finishes
            self._cleanup_upload(saved_file_path)

    def _save_timing_data(self, base_filename: str, timing_metadata: "TimingMetadata") -> None:
        """Save timing metadata as JSON file."""
        timing_json = {
            "total_duration": timing_metadata.total_duration,
            "audio_files": timing_metadata.audio_files,
            "text_segments": [
                {
                    "text": clean_text_for_display(segment.text),
                    "start_time": segment.start_time,
                    "duration": segment.duration,
                    "segment_type": segment.segment_type,
                    "chunk_index": segment.chunk_index,
                    "sentence_index": segment.sentence_index,
                }
                for segment in timing_metadata.text_segments
            ],
        }

        timing_filename = timing_json_filename(base_filename)
        timing_path = Path(self.config.files.audio_folder) / timing_filename

        try:
            with timing_path.open("w", encoding="utf-8") as f:
                json.dump(timing_json, f, indent=2, ensure_ascii=False)

            logger.info("Saved timing data: %s", timing_filename)
        except Exception as e:
            logger.error("Failed to save timing data: %s", e)

    @staticmethod
    def _cleanup_upload(saved_file_path: str) -> None:
        """Remove the uploaded file after processing completes or is cancelled."""
        if not saved_file_path:
            return
        try:
            path = Path(saved_file_path)
            if path.exists():
                path.unlink()
                logger.debug("Cleaned up upload file: %s", saved_file_path)
        except Exception:
            logger.warning("Failed to clean up upload file: %s", saved_file_path, exc_info=True)
