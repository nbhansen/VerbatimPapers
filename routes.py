"""Flask route handlers with proper dependency injection.

This module provides all route handlers; they reach their dependencies through
the Services instance stored on the app, eliminating global state.
"""

from concurrent.futures import Future
from functools import partial
import hmac
import json
import logging
import math
from pathlib import Path
from typing import Any
import uuid

from flask import Flask, Response, current_app, jsonify, render_template, request, send_from_directory
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from application.composition import Services
from application.config.system_config import SystemConfig
from application.services.document_service import timing_json_filename
from application.services.error_formatting import (
    get_processing_stage_error,
)
from application.services.progress_store import ThreadSafeProgressStore
from domain.models import PageRange, TimedAudioResult
from utils import (
    allowed_file,
    parse_page_range_from_form,
)

logger = logging.getLogger(__name__)


def error_page(message: str, status: int) -> tuple[str, int]:
    """Render a user-facing error page; Jinja autoescapes the message."""
    return render_template("error.html", error_message=message), status


def background_process_document(
    operation_id: str,
    request_form: dict[str, str],
    saved_file_path: str,
    original_filename: str,
    services: Services,
    app: Flask,
) -> None:
    """Background processing function that delegates to DocumentProcessingService.

    Args:
        operation_id: Unique ID for this operation
        request_form: Form data as dict
        saved_file_path: Path where the uploaded file was saved
        original_filename: Original filename from upload
        services: The app's services
        app: Flask app instance
    """
    # Use the Flask app context for this thread
    with app.app_context():
        logger.debug("Background thread started for operation %s, file: %s", operation_id[:8], original_filename)
        try:
            # Delegate all orchestration logic to the document service
            services.document_service.process_document_background(
                operation_id=operation_id,
                request_form=request_form,
                saved_file_path=saved_file_path,
                original_filename=original_filename,
            )

        except Exception as e:
            logger.exception("Background process wrapper failed: %s", e)
            enhanced_error = get_processing_stage_error("processing", e, original_filename)
            get_progress_store().update(
                operation_id, "error", 0, "Processing failed unexpectedly", is_error=True, error_message=enhanced_error
            )


def handle_background_completion(
    future: Future[None], operation_id: str, progress_store: ThreadSafeProgressStore
) -> None:
    """Surface exceptions that escaped background_process_document's own handling.

    Without this, an exception raised outside the wrapper's try block would sit
    silently on the discarded Future while the operation hangs at its last
    progress state.
    """
    if future.cancelled():
        return
    exc = future.exception()
    if exc is None:
        return
    logger.error("Uncaught error in background operation %s", operation_id[:8], exc_info=exc)
    progress_store.update(
        operation_id, "error", 0, "Processing failed unexpectedly", is_error=True, error_message=str(exc)
    )


def get_services() -> Services:
    """Get the app's services from the Flask app config."""
    services: Services = current_app.config["SERVICES"]
    return services


def get_app_config() -> SystemConfig:
    """Get the app's configuration."""
    return get_services().config


def get_progress_store() -> ThreadSafeProgressStore:
    """Get the progress store."""
    return get_services().progress_store


def register_routes(app: Flask) -> None:
    """Register all routes with the Flask app."""

    @app.route("/")
    def index() -> str:
        config = get_app_config()
        return render_template(
            "index.html",
            llm_available=get_services().llm_provider is not None,
            max_file_size_mb=config.files.max_file_size_mb,
        )

    @app.route("/favicon.ico")
    def favicon() -> tuple[str, int]:
        """Return a simple response for favicon requests to avoid 404 errors."""
        return "", 204

    @app.route("/audio_outputs/<filename>")
    def serve_audio(filename: str) -> Response:
        return send_from_directory(app.config["AUDIO_FOLDER"], filename)

    @app.route("/read-along/<base_filename>")
    def read_along_view(base_filename: str) -> str | tuple[str, int]:
        """Serve the read-along interface for a timed conversion.

        Keyed by the conversion's base filename; the audio file is the one the
        timing data records, so no filename is derived from another.
        """
        timing_path = Path(app.config["AUDIO_FOLDER"]) / timing_json_filename(base_filename)
        if not timing_path.exists():
            return error_page("Timing data not found. This file was not processed with read-along support.", 404)

        try:
            with timing_path.open(encoding="utf-8") as f:
                timing_json = json.load(f)
        except (OSError, ValueError) as e:
            logger.error("Error loading timing data: %s", str(e))
            return error_page("Failed to load timing data.", 500)

        audio_files = timing_json.get("audio_files") or []
        audio_filename = audio_files[0] if audio_files else None
        if (
            not isinstance(audio_filename, str)
            or Path(audio_filename).name != audio_filename
            or not (Path(app.config["AUDIO_FOLDER"]) / audio_filename).exists()
        ):
            return error_page("Audio file not found.", 404)

        return render_template(
            "read_along.html",
            audio_filename=audio_filename,
            base_filename=base_filename,
            timing_data=timing_json.get("text_segments", []),
        )

    @app.route("/api/progress/<operation_id>")
    def get_progress_status(operation_id: str) -> Response | tuple[Response, int]:
        """Get current progress status for an operation."""
        progress = get_progress_store().get(operation_id)
        if progress is None:
            return jsonify({"error": "Operation not found"}), 404

        return jsonify(
            {
                "operation_id": progress.operation_id,
                "stage": progress.stage,
                "percentage": progress.percentage,
                "message": progress.message,
                "complete": progress.is_complete,
                "error": progress.is_error,
                "error_message": progress.error_message,
                "cancelled": progress.cancelled,
                "result_data": progress.result_data,
            }
        )

    @app.route("/api/cancel/<operation_id>", methods=["POST"])
    def cancel_processing_operation(operation_id: str) -> Response | tuple[Response, int]:
        """Cancel a running operation."""
        progress = get_progress_store().get(operation_id)
        if progress is None:
            return jsonify({"error": "Operation not found"}), 404

        if progress.is_complete:
            return jsonify({"message": "Operation already completed", "cancelled": False}), 200

        get_progress_store().cancel(operation_id)
        logger.info("Cancellation requested for operation: %s", operation_id[:8])
        return jsonify({"message": "Cancellation requested", "cancelled": True})

    @app.route("/result/<operation_id>")
    def show_result(operation_id: str) -> str | tuple[str, int]:
        """Show results for a completed operation."""
        progress = get_progress_store().get(operation_id)
        if progress is None:
            return error_page("Operation not found.", 404)

        # Failed operations are never marked complete, so check for errors first
        if progress.is_error or progress.error_message:
            return error_page(f"Processing failed: {progress.error_message or 'Unknown error'}", 500)

        if progress.cancelled:
            return error_page("Operation was cancelled.", 400)

        if not progress.is_complete:
            return error_page("Operation not yet complete.", 400)

        if not progress.result_data:
            return error_page("No result data available.", 500)

        # Extract result data
        result_data = progress.result_data
        base_filename = result_data.get("base_filename", "unknown")
        original_filename = result_data.get("original_filename", "unknown")
        audio_result = result_data.get("audio_result")

        if not audio_result:
            return error_page("No processing result available.", 500)

        # Page range travels through result_data so the result page reflects the selection
        page_range = result_data.get("page_range") or PageRange(start_page=None, end_page=None)

        return render_upload_result(audio_result, original_filename, base_filename, page_range)

    def _validate_upload_request() -> tuple[FileStorage | None, str | None]:
        """Common validation for upload endpoints.

        Returns:
            (file, error_message) - file if valid, error message if invalid
        """
        logger.debug("Validating upload request")
        if "pdf_file" not in request.files:
            return None, "No file was uploaded. Please select a PDF file and try again."

        file = request.files["pdf_file"]
        if not file.filename or file.filename == "":
            return None, "No file was selected. Please choose a PDF file to upload."
        elif not allowed_file(file.filename):
            return None, f"File '{file.filename}' is not a PDF. Only PDF files are supported for audio conversion."

        return file, None

    @app.route("/upload", methods=["POST"])
    def upload_file() -> str | tuple[str, int]:
        """Queue a conversion; every conversion produces audio plus a read-along timeline."""
        file, error = _validate_upload_request()
        if error or file is None:
            return error_page(error or "No file provided.", 400)

        # Reject bad page ranges before the file is saved and queued; the same
        # parse in the background thread would surface as a generic failure.
        try:
            parse_page_range_from_form(request.form)
        except ValueError as e:
            return error_page(f"Invalid page range: {e}", 400)

        # Generate unique operation ID
        operation_id = str(uuid.uuid4())

        # Initialize progress tracking
        get_progress_store().update(operation_id, "starting", 0, "Upload received, starting processing...")

        # Save file first before passing to thread (FileStorage closes after request)
        config = get_app_config()
        if not file.filename:
            return error_page("No filename provided.", 400)
        # secure_filename can strip a hostile name down to nothing
        original_filename = secure_filename(file.filename) or "upload.pdf"
        saved_path = Path(config.files.upload_folder) / f"{operation_id}_{original_filename}"
        file.save(str(saved_path))
        logger.info("Upload received: %s (op=%s)", original_filename, operation_id[:8])
        logger.debug("Saved upload to: %s", saved_path)

        # Convert form data to dict (Flask objects don't work in threads)
        form_data = dict(request.form)

        # Submit to the bounded worker pool; uploads beyond the concurrency cap queue up
        services = get_services()
        future = services.background_executor.submit(
            background_process_document,
            operation_id,
            form_data,
            str(saved_path),
            original_filename,
            services,
            current_app._get_current_object(),  # type: ignore[attr-defined]
        )
        future.add_done_callback(
            partial(handle_background_completion, operation_id=operation_id, progress_store=services.progress_store)
        )
        logger.info("Background processing queued: op=%s", operation_id[:8])

        # Return processing page immediately
        return render_template("processing.html", operation_id=operation_id)

    def _require_admin() -> tuple[Response, int] | None:
        """Gate /admin endpoints: 404 while disabled, 403 on missing/wrong token.

        Returning 404 when disabled hides the endpoints' existence, matching the
        default config where the server binds to non-localhost hosts at will.
        """
        admin = get_app_config().admin
        if not admin.enabled:
            return jsonify({"error": "Not found"}), 404
        if admin.token:
            provided = request.headers.get("X-Admin-Token", "")
            if not hmac.compare_digest(provided.encode(), admin.token.encode()):
                return jsonify({"error": "Forbidden"}), 403
        return None

    @app.route("/admin/file_stats")
    def get_file_stats() -> Response | tuple[Response, int]:
        """Get file management statistics (admin endpoint)."""
        denied = _require_admin()
        if denied is not None:
            return denied

        try:
            audio_dir = Path(app.config["AUDIO_FOLDER"])
            if not audio_dir.exists():
                return jsonify({"error": "Audio directory not found"}), 404

            files = list(audio_dir.iterdir())
            total_size = sum(f.stat().st_size for f in files if f.is_file())

            stats = {
                "total_files": len(files),
                "total_size_mb": total_size / (1024 * 1024),
                "directory": str(audio_dir),
                "cleanup_enabled": get_app_config().cleanup.enabled,
            }
            return jsonify(stats)

        except Exception as e:
            logger.error("Admin file_stats error: %s", str(e))
            return jsonify({"error": str(e)}), 500

    @app.route("/admin/cleanup", methods=["POST"])
    def manual_cleanup() -> Response | tuple[Response, int]:
        """Trigger manual file cleanup (admin endpoint)."""
        denied = _require_admin()
        if denied is not None:
            return denied

        try:
            max_age_hours = float(request.form.get("max_age_hours", 24.0))
        except ValueError:
            return jsonify({"error": "max_age_hours must be a number"}), 400
        if not math.isfinite(max_age_hours) or max_age_hours <= 0:
            return jsonify({"error": "max_age_hours must be a positive number"}), 400

        try:
            cleanup_result = get_services().file_manager.cleanup_old_files(max_age_hours)
            if cleanup_result.is_failure or cleanup_result.value is None:
                return jsonify({"error": str(cleanup_result.error)}), 500

            stats = cleanup_result.value
            result = {
                "files_removed": stats.files_removed,
                "bytes_freed": stats.bytes_freed,
                "mb_freed": stats.bytes_freed / (1024 * 1024),
                "errors": list(stats.errors),
                "max_age_hours": max_age_hours,
            }
            return jsonify(result)

        except Exception as e:
            logger.error("Admin cleanup error: %s", str(e))
            return jsonify({"error": str(e)}), 500


def render_upload_result(
    result: TimedAudioResult | None,
    original_filename: str,
    base_filename_no_ext: str,
    page_range: PageRange,
) -> str:
    """Render the result template with appropriate parameters."""
    if result is None:
        return render_template("error.html", error_message="Processing failed - no result returned")

    display_filename = original_filename
    if not page_range.is_full_document():
        display_filename += f" (pages {page_range.start_page or 1}-{page_range.end_page or 'end'})"

    # Saved downloads drop the operation id that keeps the stored files unique
    stem = Path(original_filename).stem
    download_names = {
        name: name.replace(base_filename_no_ext, stem, 1)
        for name in [*result.audio_files, *([result.combined_mp3] if result.combined_mp3 else [])]
    }

    template_params: dict[str, Any] = {
        "download_names": download_names,
        "combined_mp3_file": result.combined_mp3,
        "original_filename": display_filename,
        "tts_engine": get_app_config().tts.engine.value,
    }

    # Results stored before every conversion carried timing have no read-along view
    template_params["has_timing_data"] = result.timing_data is not None
    template_params["base_filename"] = base_filename_no_ext

    return render_template("result.html", **template_params)
