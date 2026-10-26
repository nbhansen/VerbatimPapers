# tests/unit/test_system_config_tdd.py
"""TDD tests for SystemConfig - comprehensive coverage following red-green-refactor cycle.

Tests configuration loading, validation, and error handling without external dependencies.
"""

from unittest.mock import patch

import pytest

from application.config.app_configs import FlaskConfig
from application.config.file_configs import FileCleanupConfig, FileConfig
from application.config.processing_configs import LLMConfig, OCRConfig, PerformanceConfig, TextProcessingConfig
from application.config.system_config import SystemConfig
from domain.config.tts_config import GeminiConfig, PiperConfig, TTSConfig, TTSEngine


class TestTTSEngineEnum:
    """TDD tests for TTSEngine enumeration."""

    def test_tts_engine_enum_values(self):
        """Should define correct TTS engine values."""
        assert TTSEngine.PIPER.value == "piper"
        assert TTSEngine.GEMINI.value == "gemini"

    def test_tts_engine_enum_creation_from_string(self):
        """Should create enum from valid string values."""
        assert TTSEngine("piper") == TTSEngine.PIPER
        assert TTSEngine("gemini") == TTSEngine.GEMINI

    def test_tts_engine_enum_invalid_value_raises_error(self):
        """Should raise ValueError for invalid TTS engine values."""
        with pytest.raises(ValueError, match="'invalid_engine' is not a valid TTSEngine"):
            TTSEngine("invalid_engine")

    def test_tts_engine_enum_case_sensitivity(self):
        """Should be case-sensitive for enum values."""
        with pytest.raises(ValueError, match="'PIPER' is not a valid TTSEngine"):
            TTSEngine("PIPER")
        with pytest.raises(ValueError, match="'Gemini' is not a valid TTSEngine"):
            TTSEngine("Gemini")


class TestSystemConfigCreation:
    """TDD tests for SystemConfig basic creation and defaults."""

    def test_system_config_minimal_creation(self):
        """Should create SystemConfig with minimal required parameters."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.PIPER),
            files=FileConfig(),
            cleanup=FileCleanupConfig(),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
        )

        assert config.tts.engine == TTSEngine.PIPER
        assert config.files.upload_folder == "uploads"
        assert config.files.audio_folder == "audio_outputs"
        assert config.text_processing.enable_cleaning is True

    def test_system_config_with_custom_values(self):
        """Should create SystemConfig with custom values."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.GEMINI),
            files=FileConfig(
                upload_folder="custom_uploads",
                audio_folder="custom_audio",
            ),
            cleanup=FileCleanupConfig(),
            text_processing=TextProcessingConfig(
                enable_cleaning=False,
            ),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(api_key="test-key"),
        )

        assert config.tts.engine == TTSEngine.GEMINI
        assert config.files.upload_folder == "custom_uploads"
        assert config.files.audio_folder == "custom_audio"
        assert config.text_processing.enable_cleaning is False


class TestSystemConfigValidationTDD:
    """TDD tests for SystemConfig validation logic."""

    def test_validate_gemini_requires_api_key(self):
        """Should require API key for Gemini TTS engine."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.GEMINI),
            files=FileConfig(),
            cleanup=FileCleanupConfig(),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(api_key=None),
        )

        with pytest.raises(ValueError, match="GOOGLE_AI_API_KEY is required when TTS_ENGINE=gemini"):
            config.validate()

    def test_validate_gemini_rejects_placeholder_api_key(self):
        """Should reject placeholder API key for Gemini."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.GEMINI),
            files=FileConfig(),
            cleanup=FileCleanupConfig(),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(api_key="YOUR_GOOGLE_AI_API_KEY"),
        )

        with pytest.raises(ValueError, match="Please set a valid GOOGLE_AI_API_KEY"):
            config.validate()

    def test_validate_gemini_accepts_valid_api_key(self):
        """Should accept valid API key for Gemini."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.GEMINI),
            files=FileConfig(),
            cleanup=FileCleanupConfig(),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(api_key="valid_api_key_12345"),
            piper=PiperConfig(),
        )

        # Should not raise exception
        config.validate()

    def test_validate_piper_requires_model_name(self):
        """Should require model name for Piper TTS engine."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.PIPER),
            files=FileConfig(),
            cleanup=FileCleanupConfig(),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(),
            piper=PiperConfig(model_name=""),
        )

        with pytest.raises(ValueError, match="PIPER_MODEL_NAME cannot be empty when using Piper TTS"):
            config.validate()

    def test_validate_piper_accepts_valid_model_name(self):
        """Should accept valid model name for Piper."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.PIPER),
            files=FileConfig(),
            cleanup=FileCleanupConfig(),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(),
            piper=PiperConfig(model_name="en_US-lessac-medium"),
        )

        # Should not raise exception
        config.validate()

    def test_validate_folder_paths_cannot_be_empty(self):
        """Should test that validation works with whitespace-only paths but allows empty paths."""
        # Test whitespace-only paths that should fail validation
        whitespace_test_cases = [
            ("upload_folder", "   "),  # whitespace only
            ("audio_folder", "\t\n"),  # whitespace only
            ("download_dir", "  "),  # whitespace only for piper download_dir
        ]

        for folder_attr, invalid_path in whitespace_test_cases:
            # Test creation with invalid path should be allowed but validation should fail
            if folder_attr == "download_dir":
                config = SystemConfig(
                    tts=TTSConfig(engine=TTSEngine.PIPER),
                    files=FileConfig(),
                    cleanup=FileCleanupConfig(),
                    text_processing=TextProcessingConfig(),
                    performance=PerformanceConfig(),
                    flask=FlaskConfig(),
                    ocr=OCRConfig(),
                    llm=LLMConfig(model_name="gemini-1.5-flash"),
                    gemini=GeminiConfig(),
                    piper=PiperConfig(download_dir=invalid_path),
                )
            else:
                # For file config attributes - create explicit FileConfig based on folder_attr
                if folder_attr == "upload_folder":
                    file_config = FileConfig(upload_folder=invalid_path)
                elif folder_attr == "audio_folder":
                    file_config = FileConfig(audio_folder=invalid_path)
                else:
                    raise ValueError(f"Unknown folder attribute: {folder_attr}")

                config = SystemConfig(
                    tts=TTSConfig(engine=TTSEngine.PIPER),
                    files=file_config,
                    cleanup=FileCleanupConfig(),
                    text_processing=TextProcessingConfig(),
                    performance=PerformanceConfig(),
                    flask=FlaskConfig(),
                    ocr=OCRConfig(),
                    llm=LLMConfig(model_name="gemini-1.5-flash"),
                    gemini=GeminiConfig(),
                    piper=PiperConfig(),
                )

            # Validation should catch whitespace-only paths
            with pytest.raises(ValueError, match="cannot be empty or whitespace"):
                config.validate()

        # Test that empty strings are allowed (validation skips them)
        empty_test_cases = [
            ("upload_folder", ""),
            ("audio_folder", ""),
            ("download_dir", ""),  # Test piper download_dir
        ]

        for folder_attr, empty_path in empty_test_cases:
            if folder_attr == "download_dir":
                config = SystemConfig(
                    tts=TTSConfig(engine=TTSEngine.PIPER),
                    files=FileConfig(),
                    cleanup=FileCleanupConfig(),
                    text_processing=TextProcessingConfig(),
                    performance=PerformanceConfig(),
                    flask=FlaskConfig(),
                    ocr=OCRConfig(),
                    llm=LLMConfig(model_name="gemini-1.5-flash"),
                    gemini=GeminiConfig(),
                    piper=PiperConfig(download_dir=empty_path),
                )
            else:
                # For file config attributes - create explicit FileConfig based on folder_attr
                if folder_attr == "upload_folder":
                    file_config = FileConfig(upload_folder=empty_path)
                elif folder_attr == "audio_folder":
                    file_config = FileConfig(audio_folder=empty_path)
                else:
                    raise ValueError(f"Unknown folder attribute: {folder_attr}")

                config = SystemConfig(
                    tts=TTSConfig(engine=TTSEngine.PIPER),
                    files=file_config,
                    cleanup=FileCleanupConfig(),
                    text_processing=TextProcessingConfig(),
                    performance=PerformanceConfig(),
                    flask=FlaskConfig(),
                    ocr=OCRConfig(),
                    llm=LLMConfig(model_name="gemini-1.5-flash"),
                    gemini=GeminiConfig(),
                    piper=PiperConfig(),
                )

            # Empty paths should pass validation (they are skipped)
            config.validate()  # Should not raise

    def test_validate_file_cleanup_settings_when_enabled(self):
        """Should validate file cleanup settings when cleanup is enabled."""
        # Test valid file cleanup settings
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.PIPER),
            files=FileConfig(),
            cleanup=FileCleanupConfig(enabled=True, max_file_age_hours=24.0, auto_cleanup_interval_hours=6.0),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(),
            piper=PiperConfig(),
        )
        config.validate()  # Should not raise

        # Test invalid max_file_age_hours - create new config with invalid value
        config_invalid_age = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.PIPER),
            files=FileConfig(),
            cleanup=FileCleanupConfig(enabled=True, max_file_age_hours=0.0, auto_cleanup_interval_hours=6.0),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(),
            piper=PiperConfig(),
        )
        with pytest.raises(ValueError, match="MAX_FILE_AGE_HOURS must be positive"):
            config_invalid_age.validate()

        # Test invalid auto_cleanup_interval_hours - create new config with invalid value
        config_invalid_interval = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.PIPER),
            files=FileConfig(),
            cleanup=FileCleanupConfig(enabled=True, max_file_age_hours=24.0, auto_cleanup_interval_hours=-1.0),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(),
            piper=PiperConfig(),
        )
        with pytest.raises(ValueError, match="AUTO_CLEANUP_INTERVAL_HOURS must be positive"):
            config_invalid_interval.validate()

    def test_validate_file_cleanup_settings_when_disabled(self):
        """Should skip file cleanup validation when cleanup is disabled."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.PIPER),
            files=FileConfig(),
            cleanup=FileCleanupConfig(
                enabled=False,
                max_file_age_hours=0.0,  # Invalid values
                auto_cleanup_interval_hours=-1.0,
            ),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(),
            piper=PiperConfig(),
        )

        # Should not raise exception since cleanup is disabled
        config.validate()


class TestSystemConfigHelperMethodsTDD:
    """TDD tests for SystemConfig helper methods."""

    def test_get_piper_config_creates_correct_config(self):
        """Should create proper Piper configuration object."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.PIPER),
            files=FileConfig(),
            cleanup=FileCleanupConfig(),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(),
            piper=PiperConfig(model_name="en_US-amy-high", download_dir="/custom/models", length_scale=1.2),
        )

        result = config.piper  # Direct access to piper config

        # Should return a config object with correct values
        assert result is not None
        assert hasattr(result, "model_name")
        assert hasattr(result, "download_dir")
        assert hasattr(result, "length_scale")
        assert result.model_name == "en_US-amy-high"
        assert result.download_dir == "/custom/models"
        assert result.length_scale == 1.2

    def test_log_summary_displays_configuration_info(self):
        """Should log comprehensive configuration summary."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.GEMINI),
            files=FileConfig(upload_folder="test_uploads", audio_folder="test_audio"),
            cleanup=FileCleanupConfig(enabled=True, max_file_age_hours=48.0, auto_cleanup_interval_hours=12.0),
            text_processing=TextProcessingConfig(enable_cleaning=True),
            performance=PerformanceConfig(audio_concurrent_chunks=8),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(api_key="test_key", voice_name="Charon"),
            piper=PiperConfig(),
        )

        # Capture log output
        import logging

        mock_logger = logging.getLogger("test_log_summary")
        with patch.object(mock_logger, "info") as mock_info:
            config.log_summary(mock_logger)

            # Verify key information is logged
            logged_text = mock_info.call_args[0][0]

            assert "TTS Engine: gemini" in logged_text
            assert "Text Cleaning: Enabled" in logged_text
            assert "Audio Concurrent Chunks: 8" in logged_text
            assert "Upload Folder: test_uploads" in logged_text
            assert "Audio Folder: test_audio" in logged_text
            assert "File Cleanup: Enabled" in logged_text
            assert "Max File Age: 48.0 hours" in logged_text
            assert "Gemini API Key: Set" in logged_text
            assert "Gemini Voice: Charon" in logged_text

    def test_log_summary_handles_missing_api_key(self):
        """Should show 'Missing' for unset API key."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.GEMINI),
            files=FileConfig(),
            cleanup=FileCleanupConfig(),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(api_key=None),
            piper=PiperConfig(),
        )

        import logging

        mock_logger = logging.getLogger("test_log_summary")
        with patch.object(mock_logger, "info") as mock_info:
            config.log_summary(mock_logger)

            logged_text = mock_info.call_args[0][0]
            assert "Gemini API Key: Missing" in logged_text

    def test_log_summary_shows_piper_specific_info(self):
        """Should show Piper-specific configuration when using Piper."""
        config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.PIPER),
            files=FileConfig(),
            cleanup=FileCleanupConfig(),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="gemini-1.5-flash"),
            gemini=GeminiConfig(),
            piper=PiperConfig(
                model_name="en_US-lessac-high",
                download_dir="/custom/piper/models",
            ),
        )

        import logging

        mock_logger = logging.getLogger("test_log_summary")
        with patch.object(mock_logger, "info") as mock_info:
            config.log_summary(mock_logger)

            logged_text = mock_info.call_args[0][0]
            assert "TTS Engine: piper" in logged_text
            assert "Piper Model: en_US-lessac-high" in logged_text
            assert "Piper Models Dir: /custom/piper/models" in logged_text
