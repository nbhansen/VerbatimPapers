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
