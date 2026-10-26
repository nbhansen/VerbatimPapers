"""Application service for document processing orchestration.

This service decouples the web layer from the core domain logic by providing
a unified interface for processing documents into audio.
"""

import contextlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.config.system_config import SystemConfig
    from application.container.service_container import ServiceContainer
    from domain.models import TimingMetadata

from application.services.error_formatting import clean_text_for_display, get_processing_stage_error
from application.services.progress_store import ThreadSafeProgressStore
from domain.interfaces import IAudioEngine, IDocumentEngine, IFileManager, ITextPipeline
from domain.models import ProcessingRequest
from utils import parse_page_range_from_form, parse_plain_english_from_form

logger = logging.getLogger(__name__)


def timing_json_filename(base_filename: str) -> str:
    """Name of the read-along timing file saved for a conversion's base filename."""
    return f"{base_filename}_timing.json"


class DocumentProcessingService:
    """Service for orchestrating document-to-audio conversion pipeline."""

    def __init__(
        self,
        service_container: "ServiceContainer",
        config: "SystemConfig",
        progress_store: ThreadSafeProgressStore,
    ):
        self.container = service_container
        self.config = config
        self.progress_store = progress_store

    def process_document_background(
        self,
        operation_id: str,
        request_form: dict[str, str],
        saved_file_path: str,
        original_filename: str,
        enable_timing: bool,
    ) -> None:
        """Process a document in the background with progress reporting."""
        try:
            logger.info("Starting document processing for operation %s, file: %s", operation_id, original_filename)
            self.progress_store.update(operation_id, "starting", 5, "Initializing document processing...")

            if self.progress_store.is_cancelled(operation_id):
                return

            # 1. Validate and prepare file info (10%)
            self.progress_store.update(operation_id, "validating", 10, "Validating uploaded file...")

            # Simplified file info extraction
            base_filename = Path(original_filename).stem
            page_range = parse_page_range_from_form(request_form)
            enable_plain_english = parse_plain_english_from_form(request_form)

            if self.progress_store.is_cancelled(operation_id):
                return

            # 2. Execute processing (20-95%)
            document_engine = self.container.get(IDocumentEngine)
            audio_engine = self.container.get(IAudioEngine)
            text_pipeline = self.container.get(ITextPipeline)
            logger.debug(
                "Services resolved: document_engine=%s, audio_engine=%s, text_pipeline=%s",
                type(document_engine).__name__,
                type(audio_engine).__name__,
                type(text_pipeline).__name__,
            )

            # Override timing for Gemini TTS (as per original logic in routes.py)
            enable_timing = enable_timing and self.config.tts.engine.value != "gemini"

            # Create custom text pipeline if plain English is requested
            if enable_plain_english:
                text_pipeline = self.container.get("plain_english_text_pipeline")
                logger.info("Using TextPipeline with plain English conversion enabled")

            self.progress_store.update(operation_id, "processing", 20, "Starting document processing...")

            request_obj = ProcessingRequest(pdf_path=saved_file_path, output_name=base_filename, page_range=page_range)
            logger.debug(
                "Processing request: pdf=%s, output=%s, pages=%s-%s, timing=%s",
                request_obj.pdf_path,
                request_obj.output_name,
                page_range.start_page,
                page_range.end_page,
                enable_timing,
            )

            # Core processing
            processing_result = document_engine.process_document(
                request_obj, audio_engine, text_pipeline, enable_timing, self.config.text_processing.llm_chunk_size
            )

            if self.progress_store.is_cancelled(operation_id):
                return

            logger.debug("Processing result: success=%s", processing_result.is_success)
            if processing_result.is_failure:
                error_msg = str(processing_result.error)
                enhanced_error = get_processing_stage_error("audio_generation", Exception(error_msg), original_filename)
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

            # Save timing data if needed
            if enable_timing and audio_result.timing_data:
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

            # Register with file manager for cleanup if needed
            with contextlib.suppress(Exception):
                file_manager = self.container.get(IFileManager)
                if hasattr(file_manager, "schedule_cleanup"):
                    file_manager.schedule_cleanup(timing_filename, 2.0)
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
