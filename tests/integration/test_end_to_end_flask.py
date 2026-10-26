"""End-to-end integration tests for Flask PDF-to-Audio conversion.

Tests the complete workflow from PDF upload through audio generation
using real Flask test client with dependency injection.
"""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
import dataclasses
from io import BytesIO
import json
from pathlib import Path
import re
import threading
from typing import ClassVar
from unittest.mock import patch

from flask import Flask
from flask.testing import FlaskClient
import pytest

from app_factory import create_app
from application.config.app_configs import AdminConfig, FlaskConfig
from application.config.file_configs import FileCleanupConfig, FileConfig
from application.config.processing_configs import LLMConfig, OCRConfig, PerformanceConfig, TextProcessingConfig
from application.config.system_config import SystemConfig
from domain.config.tts_config import GeminiConfig, PiperConfig, TTSConfig, TTSEngine
from domain.errors import Result
from domain.interfaces import ILLMProvider
from domain.models import PageRange, SynthesizedSegment, TimedAudioResult
from routes import register_routes
from tests.test_helpers import FakeTTSEngine

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
        flask=FlaskConfig(debug=False),  # Serving process: create_app warms services up
        ocr=OCRConfig(),
        llm=LLMConfig(model_name="test-llm-model"),
        piper=PiperConfig(model_name="en_US-lessac-medium"),
        gemini=None,  # No Gemini for integration tests
    )


@pytest.fixture
def test_app(flask_test_config: SystemConfig) -> Generator[Flask, None, None]:
    """Flask app configured for integration testing."""
    # Fake TTS so startup needs no voice model and conversions run fast
    with (
        patch("application.composition.PiperTTSProvider", return_value=FakeTTSEngine()),
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


class EchoLLM(ILLMProvider):
    """Cleans by returning the text unchanged and "rewrites" by tagging it, so tests can see each stage."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_content(self, prompt: str) -> Result[str]:
        """Echo the prompt's text, marked PLAIN when it was a plain-English request."""
        self.prompts.append(prompt)
        if "Text to convert:\n" in prompt:
            return Result.success("PLAIN " + prompt.split("Text to convert:\n", 1)[1])
        return Result.success(prompt.split("Text:\n", 1)[1])


class BlockingTTSEngine(FakeTTSEngine):
    """Holds every synthesis until released, so a test can act while a conversion is mid-flight."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def synthesize(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Signal that synthesis started, then wait to be released."""
        self.started.set()
        assert self.release.wait(timeout=10), "test never released synthesis"
        return super().synthesize(text)


@contextmanager
def _running_app(
    config: SystemConfig, tts: FakeTTSEngine | None = None, llm: ILLMProvider | None = None
) -> Generator[tuple[Flask, FlaskClient], None, None]:
    """The real app with only TTS (and the LLM, if given) faked."""
    if llm is not None:
        config = dataclasses.replace(config, gemini=GeminiConfig(api_key="test-key"))
    with (
        patch("application.composition.PiperTTSProvider", return_value=tts or FakeTTSEngine()),
        patch("application.composition.GeminiLLMProvider", return_value=llm),
        patch("application.config.system_config.SystemConfig.from_yaml", return_value=config),
    ):
        app = create_app()
        app.config.update({"TESTING": True})
        register_routes(app)
        yield app, app.test_client()


def _upload(client: FlaskClient, content: bytes, filename: str, form: dict[str, str] | None = None) -> str:
    """Post an upload and return its operation id from the processing page."""
    response = client.post(
        "/upload",
        data={"pdf_file": (BytesIO(content), filename), **(form or {})},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    match = re.search(r'const operationId = "([0-9a-f-]+)"', response.get_data(as_text=True))
    assert match, "Processing page should carry the operation id"
    return match.group(1)


def _finish(app: Flask) -> None:
    """Wait for every queued conversion to finish."""
    app.config["SERVICES"].background_executor.shutdown(wait=True)


def _spoken_text(config: SystemConfig, base_filename: str) -> str:
    """Everything the read-along timeline says was spoken, in order."""
    timing_path = Path(config.files.audio_folder) / f"{base_filename}_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    return " ".join(segment["text"] for segment in timing["text_segments"])


# Distinctive phrases from pages 2 and 3 of testpdf.pdf (page 1 is an image-only cover)
PAGE_2_PHRASE = "Untangling the participation buzz"
PAGE_3_PHRASE = "place-making interventions are very often initiated"


class TestEndToEndPDFConversion:
    """End-to-end tests for complete PDF conversion workflow."""

    def test_upload_produces_mp3_timeline_and_result_page(
        self, sample_pdf_content: bytes, flask_test_config: SystemConfig
    ) -> None:
        """Upload, convert in the background, then the result page offers the MP3 and the files exist."""
        config = dataclasses.replace(flask_test_config, text_processing=TextProcessingConfig(enable_cleaning=False))
        with _running_app(config) as (app, client):
            operation_id = _upload(client, sample_pdf_content, "paper.pdf")
            _finish(app)

            progress = client.get(f"/api/progress/{operation_id}").get_json()
            assert progress["complete"] and not progress["error"], progress
            result_page = client.get(f"/result/{operation_id}").get_data(as_text=True)

        base = f"paper_{operation_id[:8]}"
        assert f"/audio_outputs/{base}.mp3" in result_page
        assert (Path(config.files.audio_folder) / f"{base}.mp3").stat().st_size > 0
        spoken = _spoken_text(config, base)
        assert PAGE_2_PHRASE in spoken and PAGE_3_PHRASE in spoken
        assert list(Path(config.files.upload_folder).iterdir()) == [], "The upload should be deleted"

    def test_page_range_converts_only_those_pages(
        self, sample_pdf_content: bytes, flask_test_config: SystemConfig
    ) -> None:
        """Pages 2-2 speak page 2 and nothing from page 3, and the result page names the range."""
        config = dataclasses.replace(flask_test_config, text_processing=TextProcessingConfig(enable_cleaning=False))
        with _running_app(config) as (app, client):
            operation_id = _upload(
                client, sample_pdf_content, "paper.pdf", {"use_page_range": "on", "start_page": "2", "end_page": "2"}
            )
            _finish(app)
            result_page = client.get(f"/result/{operation_id}").get_data(as_text=True)

        spoken = _spoken_text(config, f"paper_{operation_id[:8]}")
        assert PAGE_2_PHRASE in spoken
        assert PAGE_3_PHRASE not in spoken
        assert "paper.pdf (pages 2-2)" in result_page

    def test_page_range_past_the_end_fails_with_the_reason(
        self, sample_pdf_content: bytes, flask_test_config: SystemConfig
    ) -> None:
        """Asking for pages the document doesn't have is reported, not silently clipped."""
        with _running_app(flask_test_config) as (app, client):
            operation_id = _upload(
                client, sample_pdf_content, "paper.pdf", {"use_page_range": "on", "start_page": "2", "end_page": "9"}
            )
            _finish(app)
            progress = client.get(f"/api/progress/{operation_id}").get_json()

        assert progress["error"]
        assert progress["error_message"] == (
            "Invalid page range for 'paper.pdf': End page 9 is past the end of the document (3 pages)"
        )

    def test_unreadable_pdf_fails_at_extraction(self, flask_test_config: SystemConfig) -> None:
        """A file that isn't a PDF fails in the extraction stage, says so, and leaves no upload behind."""
        with _running_app(flask_test_config) as (app, client):
            operation_id = _upload(client, b"This is not a PDF file content", "fake.pdf")
            _finish(app)

            progress = client.get(f"/api/progress/{operation_id}").get_json()
            result_page = client.get(f"/result/{operation_id}")

        assert progress["error"] and not progress["complete"]
        assert progress["error_message"].startswith("Failed to extract text from PDF for 'fake.pdf': ")
        assert result_page.status_code == 500
        assert "Failed to extract text from PDF" in result_page.get_data(as_text=True)
        assert list(Path(flask_test_config.files.upload_folder).iterdir()) == []
        assert list(Path(flask_test_config.files.audio_folder).iterdir()) == []

    @pytest.mark.parametrize("plain_english", [True, False])
    def test_plain_english_option_changes_what_is_spoken(
        self, sample_pdf_content: bytes, flask_test_config: SystemConfig, plain_english: bool
    ) -> None:
        """Ticking "Convert to plain English" sends the cleaned text for rewriting and speaks the rewrite."""
        llm = EchoLLM()
        form = {"use_page_range": "on", "start_page": "2", "end_page": "2"}
        if plain_english:
            form["enable_plain_english"] = "on"
        with _running_app(flask_test_config, llm=llm) as (app, client):
            operation_id = _upload(client, sample_pdf_content, "paper.pdf", form)
            _finish(app)

        rewrite_prompts = [p for p in llm.prompts if "Text to convert:\n" in p]
        spoken = _spoken_text(flask_test_config, f"paper_{operation_id[:8]}")
        assert PAGE_2_PHRASE in spoken
        if plain_english:
            assert rewrite_prompts, "The rewrite prompt should be sent"
            assert spoken.startswith("PLAIN ")
        else:
            assert rewrite_prompts == []
            assert "PLAIN" not in spoken

    def test_cancel_mid_conversion(self, sample_pdf_content: bytes, flask_test_config: SystemConfig) -> None:
        """Cancelling mid-synthesis stops the work: no further chunks, no MP3, no result page, no upload left."""
        tts = BlockingTTSEngine()
        config = dataclasses.replace(
            flask_test_config,
            # Page 2 becomes several chunks, synthesized one at a time
            text_processing=TextProcessingConfig(enable_cleaning=False, audio_target_chunk_size=500),
            performance=PerformanceConfig(audio_concurrent_chunks=1),
        )
        with _running_app(config, tts=tts) as (app, client):
            operation_id = _upload(
                client, sample_pdf_content, "paper.pdf", {"use_page_range": "on", "start_page": "2", "end_page": "2"}
            )
            assert tts.started.wait(timeout=10), "Synthesis never started"

            cancel = client.post(f"/api/cancel/{operation_id}").get_json()
            tts.release.set()
            _finish(app)

            progress = client.get(f"/api/progress/{operation_id}").get_json()
            result_page = client.get(f"/result/{operation_id}")

        assert cancel["cancelled"] is True
        assert progress["cancelled"] and not progress["error"]
        assert progress["result_data"] is None
        assert result_page.status_code == 400
        assert "cancelled" in result_page.get_data(as_text=True)
        assert len(tts.generated_texts) == 1, "Only the chunk already in flight should be synthesized"
        assert list(Path(config.files.audio_folder).iterdir()) == [], "A cancelled conversion writes no audio"
        assert list(Path(config.files.upload_folder).iterdir()) == []

    @pytest.mark.parametrize("engine", [TTSEngine.PIPER, TTSEngine.GEMINI])
    def test_read_along_link_loads(
        self, sample_pdf_content: bytes, flask_test_config: SystemConfig, engine: TTSEngine
    ) -> None:
        """Every conversion's result page links to a read-along view that loads, for either engine.

        Drives the real flow with only TTS faked: upload, wait for the background
        job, follow the link on the result page. Guards against the read-along
        route and the audio engine disagreeing on file names.
        """
        config = dataclasses.replace(
            flask_test_config,
            tts=TTSConfig(engine=engine, request_delay_seconds=0.0),
            gemini=GeminiConfig(api_key="test-key") if engine == TTSEngine.GEMINI else None,
            text_processing=TextProcessingConfig(enable_cleaning=False),
        )
        with (
            patch("application.composition.PiperTTSProvider", return_value=FakeTTSEngine()),
            patch("application.composition.GeminiTTSProvider", return_value=FakeTTSEngine()),
            patch("application.config.system_config.SystemConfig.from_yaml", return_value=config),
        ):
            app = create_app()
            app.config.update({"TESTING": True})
            register_routes(app)
            client = app.test_client()

            response = client.post(
                "/upload",
                data={"pdf_file": (BytesIO(sample_pdf_content), "timing_test.pdf")},
                content_type="multipart/form-data",
            )
            assert response.status_code == 200
            match = re.search(r'const operationId = "([0-9a-f-]+)"', response.get_data(as_text=True))
            assert match, "Processing page should carry the operation id"

            # Let the background job finish before inspecting its result
            app.config["SERVICES"].background_executor.shutdown(wait=True)

            result_page = client.get(f"/result/{match.group(1)}")
            assert result_page.status_code == 200, result_page.get_data(as_text=True)
            link = re.search(r'href="(/read-along/[^"]+)"', result_page.get_data(as_text=True))
            assert link, "Result page should offer Read-Along Mode"

            read_along = client.get(link.group(1))
            assert read_along.status_code == 200, read_along.get_data(as_text=True)
            base_filename = f"timing_test_{match.group(1)[:8]}"
            assert f"/audio_outputs/{base_filename}.mp3" in read_along.get_data(as_text=True)

            # Every spoken segment is on the page, carrying the times the player highlights it by
            page = read_along.get_data(as_text=True)
            segments = json.loads(
                (Path(config.files.audio_folder) / f"{base_filename}_timing.json").read_text(encoding="utf-8")
            )["text_segments"]
            assert segments, "Timing data should contain segments"
            assert page.count('class="text-segment"') == len(segments)
            assert f'data-start="{segments[1]["start_time"]}"' in page

    def test_same_named_uploads_get_distinct_outputs(
        self, sample_pdf_content: bytes, flask_test_config: SystemConfig
    ) -> None:
        """Two concurrent conversions of identically named PDFs don't overwrite each other."""
        with (
            patch("application.composition.PiperTTSProvider", return_value=FakeTTSEngine()),
            patch("application.config.system_config.SystemConfig.from_yaml", return_value=flask_test_config),
        ):
            app = create_app()
            app.config.update({"TESTING": True})
            register_routes(app)
            client = app.test_client()

            operation_ids = []
            for _ in range(2):
                response = client.post(
                    "/upload",
                    data={"pdf_file": (BytesIO(sample_pdf_content), "paper.pdf")},
                    content_type="multipart/form-data",
                )
                match = re.search(r'const operationId = "([0-9a-f-]+)"', response.get_data(as_text=True))
                assert match
                operation_ids.append(match.group(1))

            app.config["SERVICES"].background_executor.shutdown(wait=True)

            audio_dir = Path(flask_test_config.files.audio_folder)
            mp3_links = []
            for operation_id in operation_ids:
                page = client.get(f"/result/{operation_id}").get_data(as_text=True)
                mp3 = f"paper_{operation_id[:8]}.mp3"
                assert f"/audio_outputs/{mp3}" in page, page
                assert 'download="paper.mp3"' in page, "Download name should stay human-readable"
                assert (audio_dir / mp3).exists()
                assert (audio_dir / f"paper_{operation_id[:8]}_timing.json").exists()
                mp3_links.append(mp3)
            assert mp3_links[0] != mp3_links[1]


class TestBackgroundProcessingModel:
    """Tests for the bounded worker pool and result-page state handling."""

    def test_background_executor_uses_configured_cap(self, test_app: Flask, flask_test_config: SystemConfig) -> None:
        """The app's worker pool must honor performance.max_concurrent_operations."""
        context = test_app.config["SERVICES"]
        expected = flask_test_config.performance.max_concurrent_operations
        assert context.background_executor._max_workers == expected

    def test_result_page_shows_selected_page_range(self, test_app: Flask, client: FlaskClient) -> None:
        """/result must display the page range the user selected, not the full document."""
        context = test_app.config["SERVICES"]
        context.progress_store.update(
            "op-range",
            "complete",
            100,
            "Processing complete!",
            is_complete=True,
            result_data={
                "base_filename": "doc",
                "original_filename": "doc.pdf",
                "audio_result": TimedAudioResult(audio_files=["doc.mp3"], combined_mp3="doc.mp3"),
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
        context = test_app.config["SERVICES"]
        context.progress_store.update(
            "op-full",
            "complete",
            100,
            "Processing complete!",
            is_complete=True,
            result_data={
                "base_filename": "doc",
                "original_filename": "doc.pdf",
                "audio_result": TimedAudioResult(audio_files=["doc.mp3"], combined_mp3="doc.mp3"),
            },
        )

        response = client.get("/result/op-full")

        assert response.status_code == 200
        assert "pages" not in response.get_data(as_text=True).lower().split("doc.pdf")[1][:30]

    def test_result_without_timeline_offers_no_read_along(self, test_app: Flask, client: FlaskClient) -> None:
        """A result whose timeline wasn't saved shows the MP3 but no Read-Along link to a 404."""
        test_app.config["SERVICES"].progress_store.update(
            "op-no-timing",
            "complete",
            100,
            "Processing complete!",
            is_complete=True,
            result_data={
                "base_filename": "doc",
                "original_filename": "doc.pdf",
                "audio_result": TimedAudioResult(audio_files=["doc.mp3"], combined_mp3="doc.mp3", timing_data=None),
            },
        )

        page = client.get("/result/op-no-timing").get_data(as_text=True)

        assert "/audio_outputs/doc.mp3" in page
        assert "/read-along/" not in page


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
        patch("application.composition.PiperTTSProvider", return_value=FakeTTSEngine()),
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
            response = admin_client.get("/admin/file_stats")
            assert response.status_code == 200

    def test_admin_token_rejects_missing_or_wrong_header(self, flask_test_config: SystemConfig) -> None:
        """With a token configured, requests without the right header get 403."""
        with _app_with_admin(flask_test_config, AdminConfig(enabled=True, token="hunter2")) as admin_client:  # noqa: S106
            assert admin_client.get("/admin/file_stats").status_code == 403
            wrong = admin_client.get("/admin/file_stats", headers={"X-Admin-Token": "wrong"})
            assert wrong.status_code == 403

    def test_admin_token_accepts_correct_header(self, flask_test_config: SystemConfig) -> None:
        """The configured token in X-Admin-Token unlocks the endpoints."""
        with _app_with_admin(flask_test_config, AdminConfig(enabled=True, token="hunter2")) as admin_client:  # noqa: S106
            response = admin_client.get("/admin/file_stats", headers={"X-Admin-Token": "hunter2"})
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

    @pytest.mark.parametrize(
        ("method", "path", "target"),
        [
            ("POST", "/admin/cleanup", "infrastructure.file.file_manager.FileManager.cleanup_old_files"),
            ("GET", "/admin/file_stats", "pathlib.Path.iterdir"),
        ],
    )
    def test_failure_does_not_leak_exception_text(
        self, flask_test_config: SystemConfig, method: str, path: str, target: str
    ) -> None:
        """A crash returns a fixed message; paths in the exception stay in the server log."""
        with _app_with_admin(flask_test_config, AdminConfig(enabled=True)) as admin_client:
            with patch(target, side_effect=OSError("/home/someone/private/path exploded")):
                response = admin_client.open(path, method=method)

        assert response.status_code == 500
        assert "private" not in response.get_data(as_text=True)
        assert "server log" in response.get_json()["error"]


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
        store = test_app.config["SERVICES"].progress_store
        store.update("op-failed", "error", 0, "Processing failed", is_error=True, error_message=XSS_PAYLOAD)

        response = test_app.test_client().get("/result/op-failed")

        body = response.get_data(as_text=True)
        assert response.status_code == 500
        assert "Processing failed" in body
        assert XSS_PAYLOAD not in body


class TestStartup:
    """create_app builds every service and checks the TTS engine before serving."""

    def test_unready_tts_engine_fails_startup(self, flask_test_config: SystemConfig) -> None:
        """A missing voice model is reported at startup, not on the first upload."""
        with (
            patch("application.composition.PiperTTSProvider", return_value=FakeTTSEngine(should_fail=True)),
            patch("application.config.system_config.SystemConfig.from_yaml", return_value=flask_test_config),
            pytest.raises(RuntimeError, match="TTS engine not ready"),
        ):
            create_app()

    def test_services_are_built_before_serving(self, test_app: Flask) -> None:
        """Request threads only read services that warm-up already built."""
        built = vars(test_app.config["SERVICES"])
        assert {"document_service", "tts_engine", "background_executor", "progress_store"} <= built.keys()

    def test_reloader_parent_skips_warm_up(
        self, flask_test_config: SystemConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The debug reloader's file-watching parent never loads the TTS engine."""
        monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
        config = dataclasses.replace(flask_test_config, flask=FlaskConfig(debug=True))
        with (
            patch("application.composition.PiperTTSProvider", return_value=FakeTTSEngine(should_fail=True)),
            patch("application.config.system_config.SystemConfig.from_yaml", return_value=config),
        ):
            app = create_app()

        assert "tts_engine" not in vars(app.config["SERVICES"])


class TestUploadPage:
    """The upload form only offers what this install can actually do."""

    def test_plain_english_disabled_without_api_key(self, flask_test_config: SystemConfig) -> None:
        """Without a key the option can't work, so it's disabled and the page says why."""
        with _running_app(flask_test_config) as (_, client):
            page = client.get("/").get_data(as_text=True)

        assert re.search(r'<input type="checkbox" id="enable_plain_english"[^>]*\bdisabled\b', page)
        assert 'id="llm_unavailable"' in page
        assert "secrets.google_ai_api_key" in page

    def test_plain_english_offered_with_api_key(self, flask_test_config: SystemConfig) -> None:
        """With a key the option is live and no warning is shown."""
        with _running_app(flask_test_config, llm=EchoLLM()) as (_, client):
            page = client.get("/").get_data(as_text=True)

        assert not re.search(r'<input type="checkbox" id="enable_plain_english"[^>]*\bdisabled\b', page)
        assert 'id="llm_unavailable"' not in page

    def test_size_limit_comes_from_config(self, flask_test_config: SystemConfig) -> None:
        """The browser-side check uses files.max_file_size_mb, the same limit the server enforces."""
        with _running_app(flask_test_config) as (app, client):
            page = client.get("/").get_data(as_text=True)

        assert flask_test_config.files.max_file_size_mb == 10
        assert "const maxSizeMb = 10;" in page
        assert app.config["MAX_CONTENT_LENGTH"] == 10 * 1024 * 1024
