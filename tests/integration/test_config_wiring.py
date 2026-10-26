"""Settings in config.yaml must reach the components they configure."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from application.config.system_config import SystemConfig
from application.container.service_container import ServiceContainer
from domain.audio.audio_engine import AudioEngine
from domain.document.document_engine import DocumentEngine
from domain.interfaces import IAudioEngine, IDocumentEngine
from infrastructure.llm.gemini_llm_provider import GeminiLLMProvider
from infrastructure.ocr.tesseract_ocr_provider import TesseractOCRProvider
from tests.test_helpers import FakeTTSEngine

EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "config.example.yaml"


def _load_example(tmp_path: Path, overrides: dict[str, dict[str, Any]] | None = None) -> SystemConfig:
    """Load config.example.yaml with per-section overrides applied."""
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    for section, values in (overrides or {}).items():
        data[section].update(values)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return SystemConfig.from_yaml(str(config_file))


@pytest.fixture(autouse=True)
def _in_tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run from tmp_path so relative upload/audio folders are created there."""
    monkeypatch.chdir(tmp_path)


class TestExampleConfig:
    """config.example.yaml documents only keys the app actually reads."""

    def test_every_example_key_is_read(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Loading the example logs no unknown-key warnings."""
        with caplog.at_level(logging.WARNING, logger="application.config.system_config"):
            _load_example(tmp_path)

        assert [r.getMessage() for r in caplog.records] == []

    def test_unknown_key_is_warned_about(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A typo'd key is reported rather than silently ignored."""
        with caplog.at_level(logging.WARNING, logger="application.config.system_config"):
            _load_example(tmp_path, {"ocr": {"langauge": "dan"}})

        assert [r.getMessage() for r in caplog.records] == ["Ignoring unknown config key: ocr.langauge"]


class TestSettingsReachConsumers:
    """Non-default values flow from YAML through the container to each component."""

    @pytest.fixture
    def container(self, tmp_path: Path) -> ServiceContainer:
        """Container built from the example config with non-default values."""
        config = _load_example(
            tmp_path,
            {
                "performance": {"audio_concurrent_chunks": 9},
                "text_processing": {"audio_target_chunk_size": 1234},
                "ocr": {"language": "dan", "dpi": 450},
                "llm": {"concurrent_requests": 7, "request_delay_seconds": 1.5},
                "secrets": {"google_ai_api_key": "test-key"},
            },
        )
        return ServiceContainer(config)

    def test_audio_engine_settings(self, container: ServiceContainer) -> None:
        """Audio concurrency and chunk size come from the config."""
        with patch("infrastructure.tts.piper_tts_provider.PiperTTSProvider", return_value=FakeTTSEngine()):
            engine = container.get(IAudioEngine)

        assert isinstance(engine, AudioEngine)
        assert engine.max_concurrent == 9
        assert engine.audio_target_chunk_size == 1234

    def test_ocr_settings(self, container: ServiceContainer) -> None:
        """OCR language reaches Tesseract; DPI reaches page rendering."""
        assert container.get(TesseractOCRProvider).ocr_language == "dan"

        document_engine = container.get(IDocumentEngine)
        assert isinstance(document_engine, DocumentEngine)
        assert document_engine.ocr_resolution == 450
        assert document_engine.audio_target_chunk_size == 1234

    def test_llm_rate_limits(self, container: ServiceContainer) -> None:
        """LLM concurrency and request spacing come from the llm section."""
        llm = container.get(GeminiLLMProvider)

        assert llm.max_concurrent_requests == 7
        assert llm.min_request_interval == 1.5
