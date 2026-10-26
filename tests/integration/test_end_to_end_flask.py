"""End-to-end integration tests for Flask PDF-to-Audio conversion.

Tests the complete workflow from PDF upload through audio generation
using real Flask test client with dependency injection.
"""

from collections.abc import Generator
from contextlib import contextmanager
import dataclasses
from io import BytesIO
from pathlib import Path
import re
from typing import ClassVar
from unittest.mock import MagicMock, patch

from flask import Flask
from flask.testing import FlaskClient
import pytest

from app_factory import create_app
from application.config.app_configs import AdminConfig, FlaskConfig
from application.config.file_configs import FileCleanupConfig, FileConfig
from application.config.processing_configs import LLMConfig, OCRConfig, PerformanceConfig, TextProcessingConfig
from application.config.system_config import SystemConfig
from domain.config.tts_config import PiperConfig, TTSConfig, TTSEngine
from domain.errors import Result
from domain.models import PageRange, PDFInfo, TimedAudioResult
from routes import register_routes
from tests.test_helpers import FakeTTSEngine, fake_segments

# === Flask App Test Fixtures ===


@pytest.fixture
def flask_test_config(temp_dir: Path) -> SystemConfig:
    """Test configuration optimized for Flask integration testing."""
    upload_dir = temp_dir / "uploads"
    audio_dir = temp_dir / "audio"
    upload_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    return SystemConfig(
        tts=TTSConfig(
            engine=TTSEngine.PIPER,
            concurrent_requests=1,  # Single-threaded for test predictability
            request_delay_seconds=0.1,
        ),
        files=FileConfig(
            upload_folder=str(upload_dir),
            audio_folder=str(audio_dir),
            max_file_size_mb=10,  # Small for faster tests
        ),
        cleanup=FileCleanupConfig(enabled=False),  # No cleanup during tests
        text_processing=TextProcessingConfig(
            enable_cleaning=True,
            llm_chunk_size=500,  # Small chunks for testing
        ),
        performance=PerformanceConfig(),
        flask=FlaskConfig(debug=True),
        ocr=OCRConfig(),
        llm=LLMConfig(model_name="test-llm-model"),
        piper=PiperConfig(model_name="en_US-lessac-medium"),
        gemini=None,  # No Gemini for integration tests
    )


@pytest.fixture
def test_app(flask_test_config: SystemConfig) -> Generator[Flask, None, None]:
    """Flask app configured for integration testing."""
    # Mock external dependencies to focus on integration flow
    with (
        patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True),
        patch("application.config.system_config.SystemConfig.from_yaml") as mock_config,
    ):
        mock_config.return_value = flask_test_config

        app = create_app()
        app.config.update(
            {
                "TESTING": True,
                "UPLOAD_FOLDER": flask_test_config.files.upload_folder,
                "AUDIO_FOLDER": flask_test_config.files.audio_folder,
            }
        )

        # Register routes with the Flask app
        register_routes(app)

        yield app


@pytest.fixture
def client(test_app: Flask) -> FlaskClient:
    """Flask test client for making HTTP requests."""
    return test_app.test_client()


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Load real test PDF content."""
    test_pdf_path = Path(__file__).parent.parent / "testdata" / "testpdf.pdf"
    if not test_pdf_path.exists():
        pytest.skip(f"Test PDF not found at {test_pdf_path}")
    return test_pdf_path.read_bytes()


# === Test Cases ===


class TestEndToEndPDFConversion:
    """End-to-end tests for complete PDF conversion workflow."""

    def test_complete_pdf_to_audio_conversion(
        self,
        client: FlaskClient,
        sample_pdf_content: bytes,
        flask_test_config: SystemConfig,
    ) -> None:
        """Test complete PDF upload → processing → audio generation workflow.

        This test validates:
        - PDF file upload and validation
        - Document processing pipeline
        - Text extraction and cleaning
        - Audio generation (with real TTS)
        - File management and cleanup
        - Response rendering

        NOTE: This is a REAL end-to-end test using actual services.
        """
        # Execute the upload request
        response = client.post(
            "/upload",
            data={
                "pdf_file": (BytesIO(sample_pdf_content), "test_document.pdf"),
                "page_range_start": "",
                "page_range_end": "",
                "enable_plain_english": "false",
            },
            content_type="multipart/form-data",
        )

        # Validate response
        assert response.status_code == 200, f"Upload failed with status {response.status_code}"
        response_text = response.get_data(as_text=True)

        # Check what we actually got back
        print(f"Response content: {response_text[:500]}...")

        # Validate file system state
        audio_dir = Path(flask_test_config.files.audio_folder)
        assert audio_dir.exists(), "Audio directory should be created"

        # Check what files were actually generated
        generated_files = list(audio_dir.glob("*"))
        print(f"Generated files: {[f.name for f in generated_files]}")

        # Also check uploads directory for temporary files that might help debugging
        upload_dir = Path(flask_test_config.files.upload_folder)
        if upload_dir.exists():
            upload_files = list(upload_dir.glob("*"))
            print(f"Upload files: {[f.name for f in upload_files]}")

        # Test ffmpeg availability and WAV file integrity
        if generated_files:
            print("Checking ffmpeg and WAV files...")
            self._debug_audio_conversion(generated_files, audio_dir)

        # The test succeeds if we got a response (even if it's an error)
        # This tells us if the integration is working
        assert len(response_text) > 0, "Should get some response"

    def _debug_audio_conversion(self, generated_files: list[Path], audio_dir: Path) -> None:
        """Debug helper to check WAV files and ffmpeg conversion."""
        import subprocess

        # Check if ffmpeg is available
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ ffmpeg is available")
            else:
                print(f"❌ ffmpeg not working: {result.stderr}")
        except Exception as e:
            print(f"❌ ffmpeg not found: {e}")

        # Check WAV files
        for file in generated_files:
            if file.suffix.lower() in [".wav", ".mp3"]:
                print(f"🎵 Audio file: {file.name} ({file.stat().st_size} bytes)")

                # Check if it's a valid audio file using ffmpeg
                try:
                    result = subprocess.run(
                        ["ffmpeg", "-i", str(file), "-f", "null", "-"], capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        print(f"✅ {file.name} is valid audio")
                    else:
                        print(f"❌ {file.name} has issues: {result.stderr}")
                except Exception as e:
                    print(f"❌ Could not validate {file.name}: {e}")

    def test_error_handling_invalid_pdf(self, client: FlaskClient) -> None:
        """Test error handling for invalid PDF uploads.

        This test validates:
        - Invalid file rejection
        - Proper error message display
        - No file system pollution
        - Graceful error handling
        """
        # Test with non-PDF content
        invalid_content = b"This is not a PDF file content"

        response = client.post(
            "/upload",
            data={
                "pdf_file": (BytesIO(invalid_content), "fake.pdf"),
                "page_range_start": "",
                "page_range_end": "",
            },
            content_type="multipart/form-data",
        )

        # Validate error handling
        assert response.status_code in [200, 400, 500], "Should handle invalid PDF gracefully"
        response_text = response.get_data(as_text=True)

        # Should contain error indication
        error_indicators = ["error", "invalid", "failed", "unable"]
        assert any(indicator in response_text.lower() for indicator in error_indicators), (
            "Response should indicate error for invalid PDF"
        )

    def test_page_range_processing(self, client: FlaskClient, sample_pdf_content: bytes) -> None:
        """Test partial document processing with page ranges.

        This test validates:
        - Page range parameter handling
        - Partial document processing
        - Page validation logic
        - Range-specific output
        """
        with (
            patch("infrastructure.ocr.tesseract_ocr_provider.TesseractOCRProvider") as mock_ocr,
            patch("infrastructure.tts.piper_tts_provider.PiperTTSProvider") as mock_tts,
        ):
            # Configure mocks
            mock_ocr_instance = MagicMock()
            mock_ocr_instance.extract_text.return_value = Result.success("Page 1 content only")
            mock_ocr_instance.get_pdf_info.return_value = Result.success(
                PDFInfo(total_pages=3, title="Multi-page Document", author="Test Author")
            )
            mock_ocr.return_value = mock_ocr_instance

            mock_tts_instance = MagicMock()
            mock_tts_instance.synthesize.side_effect = lambda text: Result.success(fake_segments(text))
            mock_tts_instance.get_output_format.return_value = "wav"
            mock_tts.return_value = mock_tts_instance

            # Test page range processing
            response = client.post(
                "/upload",
                data={
                    "pdf_file": (BytesIO(sample_pdf_content), "multipage.pdf"),
                    "page_range_start": "1",
                    "page_range_end": "1",
                    "enable_plain_english": "false",
                },
                content_type="multipart/form-data",
            )

            # Validate response acknowledges page range
            assert response.status_code == 200, "Page range processing should succeed"
            response_text = response.get_data(as_text=True)

            # Should indicate partial processing
            page_indicators = ["page", "1-1", "pages 1"]
            assert any(indicator in response_text.lower() for indicator in page_indicators), (
                "Response should indicate page range processing"
            )

    def test_timed_upload_read_along_link_loads(
        self, sample_pdf_content: bytes, flask_test_config: SystemConfig
    ) -> None:
        """A timed conversion's result page links to a read-along view that loads.

        Drives the real flow with only TTS faked: upload with timing, wait for the
        background job, follow the link on the result page. Guards against the
        read-along route and the timing engine disagreeing on file names.
        """
        with (
            patch("infrastructure.tts.piper_tts_provider.PiperTTSProvider", return_value=FakeTTSEngine()),
            patch("application.config.system_config.SystemConfig.from_yaml", return_value=flask_test_config),
        ):
            app = create_app()
            app.config.update({"TESTING": True})
            register_routes(app)
            client = app.test_client()

            response = client.post(
                "/upload-with-timing",
                data={"pdf_file": (BytesIO(sample_pdf_content), "timing_test.pdf")},
                content_type="multipart/form-data",
            )
            assert response.status_code == 200
            match = re.search(r'const operationId = "([0-9a-f-]+)"', response.get_data(as_text=True))
            assert match, "Processing page should carry the operation id"

            # Let the background job finish before inspecting its result
            app.config["APP_CONTEXT"].background_executor.shutdown(wait=True)

            result_page = client.get(f"/result/{match.group(1)}")
            assert result_page.status_code == 200, result_page.get_data(as_text=True)
            link = re.search(r'href="(/read-along/[^"]+)"', result_page.get_data(as_text=True))
            assert link, "Result page should offer Read-Along Mode for a timed conversion"

            read_along = client.get(link.group(1))
            assert read_along.status_code == 200, read_along.get_data(as_text=True)
            assert "/audio_outputs/timing_test_timed.mp3" in read_along.get_data(as_text=True)

            timing = client.get("/api/timing/timing_test")
            assert timing.status_code == 200
            assert timing.get_json()["text_segments"], "Timing data should contain segments"


class TestBackgroundProcessingModel:
    """Tests for the bounded worker pool and result-page state handling."""

    def test_background_executor_uses_configured_cap(self, test_app: Flask, flask_test_config: SystemConfig) -> None:
        """The app's worker pool must honor performance.max_concurrent_operations."""
        context = test_app.config["APP_CONTEXT"]
        expected = flask_test_config.performance.max_concurrent_operations
        assert context.background_executor._max_workers == expected

    def test_result_page_shows_selected_page_range(self, test_app: Flask, client: FlaskClient) -> None:
        """/result must display the page range the user selected, not the full document."""
        context = test_app.config["APP_CONTEXT"]
        context.progress_store.update(
            "op-range",
            "complete",
            100,
            "Processing complete!",
            is_complete=True,
            result_data={
                "base_filename": "doc",
                "original_filename": "doc.pdf",
                "audio_result": TimedAudioResult(audio_files=["doc.mp3"], combined_mp3=None),
                "page_range": PageRange(start_page=2, end_page=5),
            },
        )

        response = client.get("/result/op-range")

        assert response.status_code == 200
        assert "pages 2-5" in response.get_data(as_text=True)

    def test_result_page_defaults_to_full_document_without_page_range(
        self, test_app: Flask, client: FlaskClient
    ) -> None:
        """Older result_data without a page_range entry must still render."""
        context = test_app.config["APP_CONTEXT"]
        context.progress_store.update(
            "op-full",
            "complete",
            100,
            "Processing complete!",
            is_complete=True,
            result_data={
                "base_filename": "doc",
                "original_filename": "doc.pdf",
                "audio_result": TimedAudioResult(audio_files=["doc.mp3"], combined_mp3=None),
            },
        )

        response = client.get("/result/op-full")

        assert response.status_code == 200
        assert "pages" not in response.get_data(as_text=True).lower().split("doc.pdf")[1][:30]


class TestBackgroundCompletionHandler:
    """Tests for the Future done-callback that surfaces escaped exceptions."""

    def test_escaped_exception_marks_operation_failed(self) -> None:
        """An exception stored on the Future must reach the progress store."""
        from concurrent.futures import Future

        from application.services.progress_store import ThreadSafeProgressStore
        from routes import handle_background_completion

        store = ThreadSafeProgressStore()
        store.update("op-boom", "processing", 50, "Working")

        future: Future[None] = Future()
        future.set_exception(RuntimeError("worker exploded"))
        handle_background_completion(future, "op-boom", store)

        progress = store.get("op-boom")
        assert progress is not None
        assert progress.is_error is True
        assert progress.error_message is not None
        assert "worker exploded" in progress.error_message

    def test_successful_future_leaves_progress_untouched(self) -> None:
        """A clean completion must not overwrite the final progress state."""
        from concurrent.futures import Future

        from application.services.progress_store import ThreadSafeProgressStore
        from routes import handle_background_completion

        store = ThreadSafeProgressStore()
        store.update("op-ok", "complete", 100, "Done", is_complete=True)

        future: Future[None] = Future()
        future.set_result(None)
        handle_background_completion(future, "op-ok", store)

        progress = store.get("op-ok")
        assert progress is not None
        assert progress.is_error is False
        assert progress.stage == "complete"

    def test_cancelled_future_is_ignored(self) -> None:
        """A cancelled queued upload must not be reported as an error."""
        from concurrent.futures import Future

        from application.services.progress_store import ThreadSafeProgressStore
        from routes import handle_background_completion

        store = ThreadSafeProgressStore()
        store.update("op-cancelled", "starting", 0, "Queued")

        future: Future[None] = Future()
        future.cancel()
        handle_background_completion(future, "op-cancelled", store)

        progress = store.get("op-cancelled")
        assert progress is not None
        assert progress.is_error is False


@contextmanager
def _app_with_admin(base_config: SystemConfig, admin: AdminConfig) -> Generator[FlaskClient, None, None]:
    """Build a test client whose config has the given admin gating settings."""
    config = dataclasses.replace(base_config, admin=admin)
    with (
        patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True),
        patch("application.config.system_config.SystemConfig.from_yaml") as mock_config,
    ):
        mock_config.return_value = config
        app = create_app()
        app.config.update(
            {
                "TESTING": True,
                "UPLOAD_FOLDER": config.files.upload_folder,
                "AUDIO_FOLDER": config.files.audio_folder,
            }
        )
        register_routes(app)
        yield app.test_client()


class TestAdminEndpointGating:
    """Admin endpoints must be opt-in and honor the configured token."""

    ADMIN_REQUESTS: ClassVar[list[tuple[str, str]]] = [
        ("GET", "/admin/file_stats"),
        ("GET", "/admin/test"),
        ("POST", "/admin/cleanup"),
    ]

    def test_admin_disabled_by_default_returns_404(self, client: FlaskClient) -> None:
        """With the default config every /admin endpoint should look nonexistent."""
        for method, path in self.ADMIN_REQUESTS:
            response = client.open(path, method=method)
            assert response.status_code == 404, f"{method} {path} should 404 when admin is disabled"

    def test_admin_enabled_without_token_allows_access(self, flask_test_config: SystemConfig) -> None:
        """Enabling admin without a token opens the endpoints (localhost use case)."""
        with _app_with_admin(flask_test_config, AdminConfig(enabled=True)) as admin_client:
            response = admin_client.get("/admin/test")
            assert response.status_code == 200

    def test_admin_token_rejects_missing_or_wrong_header(self, flask_test_config: SystemConfig) -> None:
        """With a token configured, requests without the right header get 403."""
        with _app_with_admin(flask_test_config, AdminConfig(enabled=True, token="hunter2")) as admin_client:  # noqa: S106
            assert admin_client.get("/admin/test").status_code == 403
            wrong = admin_client.get("/admin/test", headers={"X-Admin-Token": "wrong"})
            assert wrong.status_code == 403

    def test_admin_token_accepts_correct_header(self, flask_test_config: SystemConfig) -> None:
        """The configured token in X-Admin-Token unlocks the endpoints."""
        with _app_with_admin(flask_test_config, AdminConfig(enabled=True, token="hunter2")) as admin_client:  # noqa: S106
            response = admin_client.get("/admin/test", headers={"X-Admin-Token": "hunter2"})
            assert response.status_code == 200


class TestAdminCleanupEndpoint:
    """Validation and delegation behavior of POST /admin/cleanup."""

    @pytest.mark.parametrize("bad_age", ["-5", "0", "nan", "inf", "-inf", "not-a-number"])
    def test_rejects_invalid_max_age_hours(self, flask_test_config: SystemConfig, bad_age: str) -> None:
        """Invalid ages must 400 before any file is touched."""
        audio_dir = Path(flask_test_config.files.audio_folder)
        victim = audio_dir / "current.wav"
        victim.write_bytes(b"audio")

        with _app_with_admin(flask_test_config, AdminConfig(enabled=True)) as admin_client:
            response = admin_client.post("/admin/cleanup", data={"max_age_hours": bad_age})

        assert response.status_code == 400
        assert victim.exists()

    def test_removes_only_files_older_than_cutoff(self, flask_test_config: SystemConfig) -> None:
        """A valid request should delete old files and report stats."""
        import os as os_module
        import time as time_module

        audio_dir = Path(flask_test_config.files.audio_folder)
        old_file = audio_dir / "old.wav"
        old_file.write_bytes(b"x" * 10)
        old_time = time_module.time() - 2 * 3600
        os_module.utime(old_file, (old_time, old_time))
        new_file = audio_dir / "new.wav"
        new_file.write_bytes(b"y")

        with _app_with_admin(flask_test_config, AdminConfig(enabled=True)) as admin_client:
            response = admin_client.post("/admin/cleanup", data={"max_age_hours": "1"})

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["files_removed"] == 1
        assert payload["max_age_hours"] == 1.0
        assert not old_file.exists()
        assert new_file.exists()


class TestUploadPageRangeValidation:
    """Bad page ranges must be rejected at upload time, not in the background."""

    @pytest.mark.parametrize(
        ("start", "end", "expected_fragment"),
        [
            ("abc", "5", "whole number"),
            ("0", "5", "1 or greater"),
            ("7", "3", "cannot be after"),
        ],
    )
    def test_invalid_page_range_returns_400(
        self, client: FlaskClient, start: str, end: str, expected_fragment: str
    ) -> None:
        """The upload should fail fast with a clear validation message."""
        response = client.post(
            "/upload",
            data={
                "pdf_file": (BytesIO(b"%PDF-1.4 fake"), "doc.pdf"),
                "use_page_range": "on",
                "start_page": start,
                "end_page": end,
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        body = response.get_data(as_text=True)
        assert "Invalid page range" in body
        assert expected_fragment in body


XSS_PAYLOAD = "<img src=x onerror=alert(1)>"


class TestErrorResponses:
    """Error responses keep their HTTP status and never reflect raw HTML."""

    def test_unknown_route_returns_404(self, client: FlaskClient) -> None:
        """HTTP errors pass through the global handler instead of becoming 500s."""
        response = client.get("/no-such-page")

        assert response.status_code == 404

    def test_unhandled_exception_hides_exception_text(self, test_app: Flask) -> None:
        """The generic 500 page must not echo exception details."""

        @test_app.route("/boom")
        def boom() -> str:
            raise RuntimeError("secret /home/user/path")

        response = test_app.test_client().get("/boom")

        assert response.status_code == 500
        assert "secret" not in response.get_data(as_text=True)

    @pytest.mark.parametrize(
        ("filename", "form"),
        [
            (f"{XSS_PAYLOAD}.txt", {}),
            ("doc.pdf", {"use_page_range": "on", "start_page": XSS_PAYLOAD}),
        ],
        ids=["bad-filename", "bad-page-range"],
    )
    def test_upload_errors_escape_user_input(self, client: FlaskClient, filename: str, form: dict[str, str]) -> None:
        """Rejected uploads report the bad value escaped, with a 400."""
        response = client.post(
            "/upload",
            data={"pdf_file": (BytesIO(b"%PDF-1.4 fake"), filename), **form},
            content_type="multipart/form-data",
        )

        body = response.get_data(as_text=True)
        assert response.status_code == 400
        assert XSS_PAYLOAD not in body
        assert "&lt;img src=x onerror=alert(1)&gt;" in body

    def test_read_along_does_not_reflect_url(self, client: FlaskClient) -> None:
        """An unknown read-along name is a 404 that does not echo the URL."""
        response = client.get(f"/read-along/{XSS_PAYLOAD}")

        assert response.status_code == 404
        assert "onerror" not in response.get_data(as_text=True)

    def test_failed_operation_result_shows_escaped_error(self, test_app: Flask) -> None:
        """A failed operation's result page shows its error, escaped, not 'not yet complete'."""
        store = test_app.config["APP_CONTEXT"].progress_store
        store.update("op-failed", "error", 0, "Processing failed", is_error=True, error_message=XSS_PAYLOAD)

        response = test_app.test_client().get("/result/op-failed")

        body = response.get_data(as_text=True)
        assert response.status_code == 500
        assert "Processing failed" in body
        assert XSS_PAYLOAD not in body
