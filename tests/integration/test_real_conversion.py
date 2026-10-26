"""Real conversions with nothing faked: pdfplumber, Tesseract, Piper and ffmpeg.

These are the only tests where the read-along timeline is checked against an MP3
that a real encoder wrote from real speech. LLM cleaning is off (it needs an API key).
"""

from io import BytesIO
from itertools import pairwise
import json
from pathlib import Path
import re
import shutil
import subprocess
from unittest.mock import patch

from flask import Flask
from flask.testing import FlaskClient
import pytest

from app_factory import create_app
from application.config.app_configs import FlaskConfig
from application.config.file_configs import FileCleanupConfig, FileConfig
from application.config.processing_configs import LLMConfig, OCRConfig, PerformanceConfig, TextProcessingConfig
from application.config.system_config import SystemConfig
from domain.config.tts_config import PiperConfig, TTSConfig, TTSEngine
from routes import register_routes

TEST_PDF = Path(__file__).parents[1] / "testdata" / "testpdf.pdf"
# Shared with the app's default download_dir, so the voice model is fetched at most once
PIPER_MODELS = Path(__file__).parents[2] / "piper_models"

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe") and shutil.which("tesseract")),
    reason="needs ffmpeg, ffprobe and tesseract",
)


@pytest.fixture
def real_app(temp_dir: Path) -> tuple[Flask, SystemConfig]:
    """The app exactly as it runs, with real Piper, building its services at startup."""
    config = SystemConfig(
        tts=TTSConfig(engine=TTSEngine.PIPER),
        files=FileConfig(upload_folder=str(temp_dir / "uploads"), audio_folder=str(temp_dir / "audio")),
        cleanup=FileCleanupConfig(enabled=False),
        text_processing=TextProcessingConfig(enable_cleaning=False),
        performance=PerformanceConfig(),
        flask=FlaskConfig(debug=False),
        ocr=OCRConfig(),
        llm=LLMConfig(model_name="test-llm-model"),
        piper=PiperConfig(model_name="en_US-lessac-medium", download_dir=str(PIPER_MODELS)),
    )
    with patch("application.config.system_config.SystemConfig.from_yaml", return_value=config):
        app = create_app()
    app.config.update({"TESTING": True})
    register_routes(app)
    return app, config


def _convert(app: Flask, client: FlaskClient, start_page: int, end_page: int) -> str:
    response = client.post(
        "/upload",
        data={
            "pdf_file": (BytesIO(TEST_PDF.read_bytes()), "paper.pdf"),
            "use_page_range": "on",
            "start_page": str(start_page),
            "end_page": str(end_page),
        },
        content_type="multipart/form-data",
    )
    match = re.search(r'const operationId = "([0-9a-f-]+)"', response.get_data(as_text=True))
    assert match, response.get_data(as_text=True)
    app.config["SERVICES"].background_executor.shutdown(wait=True)
    progress = client.get(f"/api/progress/{match.group(1)}").get_json()
    assert progress["complete"] and not progress["error"], progress["error_message"]
    return f"paper_{match.group(1)[:8]}"


def _mp3_duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(probe.stdout.strip())


def test_timeline_matches_the_real_mp3(real_app: tuple[Flask, SystemConfig]) -> None:
    """Page 2 becomes an MP3 whose length is the timeline's length, highlighting page 2's sentences."""
    app, config = real_app
    client = app.test_client()

    base = _convert(app, client, 2, 2)

    audio_dir = Path(config.files.audio_folder)
    timing = json.loads((audio_dir / f"{base}_timing.json").read_text(encoding="utf-8"))
    segments = timing["text_segments"]
    # MP3 framing pads by a few tens of milliseconds; anything more is drift
    assert _mp3_duration(audio_dir / f"{base}.mp3") == pytest.approx(timing["total_duration"], abs=0.15)
    assert len(segments) > 10, "Piper should give one highlight per sentence"
    assert all(b["start_time"] >= a["start_time"] + a["duration"] for a, b in pairwise(segments))
    spoken = " ".join(s["text"] for s in segments)
    assert "Untangling the participation buzz" in spoken
    assert "place-making interventions are very often initiated" not in spoken  # page 3

    read_along = client.get(f"/read-along/{base}")
    assert read_along.status_code == 200
    assert f"/audio_outputs/{base}.mp3" in read_along.get_data(as_text=True)
    assert client.get(f"/audio_outputs/{base}.mp3").status_code == 200
    assert list(Path(config.files.upload_folder).iterdir()) == []


def test_image_only_page_is_read_with_ocr(real_app: tuple[Flask, SystemConfig]) -> None:
    """The cover page has no text layer; its words reach the audio through Tesseract."""
    app, config = real_app

    base = _convert(app, app.test_client(), 1, 1)

    timing = json.loads((Path(config.files.audio_folder) / f"{base}_timing.json").read_text(encoding="utf-8"))
    spoken = " ".join(s["text"] for s in timing["text_segments"])
    assert "International Journal of CoCreation" in spoken
