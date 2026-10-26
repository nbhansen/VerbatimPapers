"""Settings in config.yaml must reach the components they configure."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from application.composition import Services
from application.config.system_config import SystemConfig
from domain.audio.audio_engine import AudioEngine
from domain.document.document_engine import DocumentEngine
from infrastructure.llm.gemini_llm_provider import GeminiLLMProvider
from infrastructure.ocr.tesseract_ocr_provider import TesseractOCRProvider
from infrastructure.tts.gemini_tts_provider import GeminiTTSProvider
from tests.test_helpers import FakeTTSEngine

EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "config.example.yaml"


def _load_example(
    tmp_path: Path, overrides: dict[str, dict[str, Any]] | None = None, drop: tuple[str, ...] = ()
) -> SystemConfig:
    """Load config.example.yaml with per-section overrides applied and "section.key" entries dropped."""
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    for section, values in (overrides or {}).items():
        data[section].update(values)
    for dotted in drop:
        section, key = dotted.split(".")
        del data[section][key]
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
    def container(self, tmp_path: Path) -> Services:
        """Container built from the example config with non-default values."""
        config = _load_example(
            tmp_path,
            {
                "performance": {"audio_concurrent_chunks": 9},
                "audio": {"sentence_gap": 0.7},
                "text_processing": {"audio_target_chunk_size": 1234},
                "ocr": {"language": "dan", "dpi": 450},
                "secrets": {"google_ai_api_key": "test-key"},
            },
        )
        return Services(config)

    def test_audio_engine_settings(self, container: Services) -> None:
        """Audio concurrency and chunk size come from the config."""
        with patch("application.composition.PiperTTSProvider", return_value=FakeTTSEngine()):
            engine = container.audio_engine

        assert isinstance(engine, AudioEngine)
        assert engine.max_concurrent == 9
        assert engine.audio_target_chunk_size == 1234

    def test_sentence_gap_reaches_assembler(self, container: Services) -> None:
        """audio.sentence_gap sets the gap the audio engine assembles with."""
        with patch("application.composition.PiperTTSProvider", return_value=FakeTTSEngine()):
            engine = container.audio_engine

        assert isinstance(engine, AudioEngine)
        assert engine.assembler.sentence_gap == 0.7

    def test_ocr_settings(self, container: Services) -> None:
        """OCR language reaches Tesseract; DPI reaches page rendering."""
        ocr = container.ocr_provider
        assert isinstance(ocr, TesseractOCRProvider)
        assert ocr.ocr_language == "dan"

        document_engine = container.document_engine
        assert isinstance(document_engine, DocumentEngine)
        assert document_engine.ocr_resolution == 450


class TestGeminiModelSeparation:
    """Text cleaning and voice generation use different Gemini models and never share one."""

    def test_llm_model_does_not_fall_back_to_tts_model(self, tmp_path: Path) -> None:
        """Without llm.model_name, cleaning uses the text default, not the speech-only TTS model."""
        config = _load_example(
            tmp_path,
            {"tts": {"engine": "gemini"}, "secrets": {"google_ai_api_key": "test-key"}},
            drop=("llm.model_name",),
        )
        container = Services(config)

        llm = container.llm_provider
        tts = container.tts_engine

        assert isinstance(llm, GeminiLLMProvider)
        assert isinstance(tts, GeminiTTSProvider)
        assert tts.model_name.endswith("-tts")
        assert not llm.model_name.endswith("-tts")
        assert llm.model_name == "gemini-2.5-flash"


class TestPlaceholderApiKey:
    """config.example.yaml ships a placeholder key; copying it unchanged means no key, not a broken one."""

    def test_placeholder_key_means_no_llm(self, tmp_path: Path) -> None:
        """With the placeholder, cleaning and plain English are off instead of failing on every call."""
        config = _load_example(tmp_path)

        assert not config.has_api_key
        assert Services(config).llm_provider is None

    def test_real_key_builds_the_llm(self, tmp_path: Path) -> None:
        """Any other key builds the text-cleaning client."""
        config = _load_example(tmp_path, {"secrets": {"google_ai_api_key": "test-key"}})

        assert config.has_api_key
        assert isinstance(Services(config).llm_provider, GeminiLLMProvider)

    def test_startup_log_says_cleaning_is_inactive(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Cleaning enabled without a key is reported at startup, not discovered from flat-sounding audio."""
        config = _load_example(tmp_path)

        with caplog.at_level(logging.INFO):
            config.log_summary()

        assert "Text Cleaning: Enabled but inactive (no secrets.google_ai_api_key)" in caplog.text
