"""DocumentProcessingService: one upload's conversion, from form data to progress and files."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from application.config.app_configs import FlaskConfig
from application.config.file_configs import FileCleanupConfig, FileConfig
from application.config.processing_configs import LLMConfig, OCRConfig, PerformanceConfig, TextProcessingConfig
from application.config.system_config import SystemConfig
from application.services.document_service import DocumentProcessingService
from application.services.progress_store import ThreadSafeProgressStore
from domain.config.tts_config import TTSConfig, TTSEngine
from domain.errors import Result, text_extraction_error
from domain.models import PageRange, TextSegment, TimedAudioResult, TimingMetadata

OPERATION_ID = "0123abcd-0000-0000-0000-000000000000"
BASE = "paper_0123abcd"


def _audio_result() -> TimedAudioResult:
    segments = [
        TextSegment(
            text="First sentence.",
            start_time=0.0,
            duration=1.0,
            segment_type="sentence",
            chunk_index=0,
            sentence_index=0,
        ),
        TextSegment(
            text="Wait... what?", start_time=1.2, duration=0.8, segment_type="sentence", chunk_index=0, sentence_index=1
        ),
    ]
    return TimedAudioResult(
        audio_files=[f"{BASE}.mp3"],
        combined_mp3=f"{BASE}.mp3",
        timing_data=TimingMetadata(total_duration=2.2, text_segments=segments, audio_files=[f"{BASE}.mp3"]),
    )


@pytest.fixture
def config(tmp_path: Path) -> SystemConfig:
    """Config writing outputs into a temp directory."""
    return SystemConfig(
        tts=TTSConfig(engine=TTSEngine.PIPER),
        files=FileConfig(upload_folder=str(tmp_path / "uploads"), audio_folder=str(tmp_path / "audio")),
        cleanup=FileCleanupConfig(enabled=False),
        text_processing=TextProcessingConfig(enable_cleaning=False, llm_chunk_size=1234),
        performance=PerformanceConfig(),
        flask=FlaskConfig(),
        ocr=OCRConfig(),
        llm=LLMConfig(model_name="test-llm-model"),
    )


@pytest.fixture
def upload(config: SystemConfig) -> Path:
    """An uploaded file waiting to be processed."""
    path = Path(config.files.upload_folder) / f"{OPERATION_ID}_paper.pdf"
    path.parent.mkdir(parents=True)
    Path(config.files.audio_folder).mkdir(parents=True)
    path.write_bytes(b"%PDF")
    return path


class Harness:
    """The service wired to a scripted document engine and a real progress store."""

    def __init__(self, config: SystemConfig, outcome: Result[TimedAudioResult] | Exception):
        self.store = ThreadSafeProgressStore()
        self.document_engine = Mock()
        if isinstance(outcome, Exception):
            self.document_engine.process_document.side_effect = outcome
        else:
            self.document_engine.process_document.return_value = outcome
        self.text_pipeline = Mock(name="text_pipeline")
        self.plain_english_pipeline = Mock(name="plain_english_pipeline")
        self.service = DocumentProcessingService(
            document_engine=self.document_engine,
            audio_engine=Mock(),
            text_pipeline=self.text_pipeline,
            plain_english_text_pipeline=self.plain_english_pipeline,
            progress_store=self.store,
            config=config,
        )

    def run(self, upload: Path, form: dict[str, str] | None = None) -> None:
        """Process the upload as the background worker would."""
        self.service.process_document_background(OPERATION_ID, form or {}, str(upload), "paper.pdf")

    def request(self) -> tuple[Any, ...]:
        """The (request, audio_engine, text_pipeline, llm_chunk_size) the engine was called with."""
        return tuple(self.document_engine.process_document.call_args.args)


class TestSuccess:
    """A successful conversion publishes its result and its read-along timeline."""

    def test_result_is_published(self, config: SystemConfig, upload: Path) -> None:
        """The result page reads everything it shows from result_data."""
        harness = Harness(config, Result.success(_audio_result()))

        harness.run(upload, {"use_page_range": "on", "start_page": "2", "end_page": "3"})

        progress = harness.store.get(OPERATION_ID)
        assert progress is not None and progress.is_complete and not progress.is_error
        assert progress.result_data is not None
        assert progress.result_data["base_filename"] == BASE
        assert progress.result_data["original_filename"] == "paper.pdf"
        assert progress.result_data["page_range"] == PageRange(2, 3)
        assert progress.result_data["audio_result"].combined_mp3 == f"{BASE}.mp3"

    def test_timing_file_matches_the_audio(self, config: SystemConfig, upload: Path) -> None:
        """Read-along loads this file, so it carries the segments exactly, text included."""
        Harness(config, Result.success(_audio_result())).run(upload)

        timing = json.loads((Path(config.files.audio_folder) / f"{BASE}_timing.json").read_text(encoding="utf-8"))
        assert timing["audio_files"] == [f"{BASE}.mp3"]
        assert timing["total_duration"] == 2.2
        assert [(s["text"], s["start_time"], s["duration"]) for s in timing["text_segments"]] == [
            ("First sentence.", 0.0, 1.0),
            ("Wait... what?", 1.2, 0.8),
        ]

    def test_request_carries_the_form_choices(self, config: SystemConfig, upload: Path) -> None:
        """Page range, output name and LLM chunk size reach the document engine."""
        harness = Harness(config, Result.success(_audio_result()))

        harness.run(upload, {"use_page_range": "on", "start_page": "4", "end_page": ""})

        request, _, pipeline, llm_chunk_size = harness.request()
        assert request.pdf_path == str(upload)
        assert request.output_name == BASE
        assert request.page_range == PageRange(start_page=4)
        assert pipeline is harness.text_pipeline
        assert llm_chunk_size == 1234

    def test_plain_english_box_selects_its_pipeline(self, config: SystemConfig, upload: Path) -> None:
        """Ticking "Convert to plain English" is what switches the text pipeline."""
        harness = Harness(config, Result.success(_audio_result()))

        harness.run(upload, {"enable_plain_english": "on"})

        assert harness.request()[2] is harness.plain_english_pipeline


class TestFailure:
    """A failed conversion tells the user which stage failed and why."""

    def test_engine_failure_is_described(self, config: SystemConfig, upload: Path) -> None:
        """The stage comes from the error, the reason from its details."""
        harness = Harness(config, Result.failure(text_extraction_error("The PDF has no pages")))

        harness.run(upload)

        progress = harness.store.get(OPERATION_ID)
        assert progress is not None and progress.is_error and not progress.is_complete
        assert progress.error_message == "Failed to extract text from PDF for 'paper.pdf': The PDF has no pages"
        assert not (Path(config.files.audio_folder) / f"{BASE}_timing.json").exists()

    def test_unexpected_exception_is_reported(self, config: SystemConfig, upload: Path) -> None:
        """A bug in a collaborator ends the operation as failed, not stuck mid-progress."""
        harness = Harness(config, RuntimeError("boom"))

        harness.run(upload)

        progress = harness.store.get(OPERATION_ID)
        assert progress is not None and progress.is_error
        assert progress.error_message is not None and "boom" in progress.error_message


class TestUploadIsDeleted:
    """The uploaded PDF never outlives its conversion, whatever the outcome."""

    @pytest.mark.parametrize(
        "outcome",
        [Result.success(_audio_result()), Result.failure(text_extraction_error("corrupt")), RuntimeError("boom")],
        ids=["success", "failure", "exception"],
    )
    def test_upload_removed(self, config: SystemConfig, upload: Path, outcome: object) -> None:
        """Uploads are deleted when processing ends, not left for the 24h sweep."""
        Harness(config, outcome).run(upload)  # type: ignore[arg-type]

        assert not upload.exists()


class TestCancellation:
    """A cancelled operation publishes nothing, even if processing finishes."""

    def test_cancel_before_start_skips_processing(self, config: SystemConfig, upload: Path) -> None:
        """An upload cancelled while queued is never converted."""
        harness = Harness(config, Result.success(_audio_result()))
        harness.store.cancel(OPERATION_ID)

        harness.run(upload)

        harness.document_engine.process_document.assert_not_called()
        assert not upload.exists()

    def test_cancel_during_processing_publishes_no_result(self, config: SystemConfig, upload: Path) -> None:
        """Cancelling mid-conversion leaves the operation cancelled, with no result page or timeline."""
        harness = Harness(config, Result.success(_audio_result()))

        def cancel_then_finish(*_args: object, **_kwargs: object) -> Result[TimedAudioResult]:
            harness.store.cancel(OPERATION_ID)
            return Result.success(_audio_result())

        harness.document_engine.process_document.side_effect = cancel_then_finish

        harness.run(upload)

        progress = harness.store.get(OPERATION_ID)
        assert progress is not None and progress.cancelled
        assert progress.result_data is None
        assert not (Path(config.files.audio_folder) / f"{BASE}_timing.json").exists()
        assert not upload.exists()
