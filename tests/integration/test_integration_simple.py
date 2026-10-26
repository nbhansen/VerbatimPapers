# tests/test_integration_simple.py
"""Simple integration test - focused on core service factory functionality."""

import tempfile
from unittest.mock import patch

from application.composition import Services
from application.config.app_configs import FlaskConfig
from application.config.file_configs import FileCleanupConfig, FileConfig
from application.config.processing_configs import LLMConfig, OCRConfig, PerformanceConfig, TextProcessingConfig
from application.config.system_config import SystemConfig
from domain.config.tts_config import GeminiConfig, TTSConfig, TTSEngine
from domain.models import PageRange, ProcessingRequest


def create_test_config() -> SystemConfig:
    """Create a minimal SystemConfig for testing."""
    return SystemConfig(
        tts=TTSConfig(engine=TTSEngine.PIPER),
        files=FileConfig(),
        cleanup=FileCleanupConfig(enabled=False),
        text_processing=TextProcessingConfig(
            enable_cleaning=False,
        ),
        performance=PerformanceConfig(),
        flask=FlaskConfig(),
        ocr=OCRConfig(),
        llm=LLMConfig(model_name="test-llm-model"),
        gemini=GeminiConfig(api_key=None),  # Provide gemini config even if no API key
    )


def test_configuration_loading():
    """Test that configuration can be loaded with minimal setup."""
    config = create_test_config()
    assert config.tts.engine == TTSEngine.PIPER
    assert config.text_processing.enable_cleaning is False
    assert config.cleanup.enabled is False
    # Test LLM model configuration
    assert config.llm.model_name == "test-llm-model"


def test_service_factory_creation():
    """Test that service factory can create all services."""
    with (
        tempfile.TemporaryDirectory(),
        # Mock piper availability to avoid infrastructure dependency
        patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True),
        patch("application.composition.PiperTTSProvider"),
    ):
        config = create_test_config()
        container = Services(config)

        # Verify all main services are available

        assert container.audio_engine is not None
        assert container.text_pipeline is not None
        assert container.document_engine is not None
        assert container.file_manager is not None


def test_basic_workflow_structure():
    """Test that basic workflow structure works with domain models."""
    # Create test request
    request = ProcessingRequest(pdf_path="test.pdf", output_name="test_output", page_range=PageRange())

    # Verify request structure
    assert request.pdf_path == "test.pdf"
    assert request.output_name == "test_output"
    assert request.page_range.is_full_document() is True


def test_error_handling_structure():
    """Test that error handling works in the service structure."""
    from domain.errors import ApplicationError, Result, configuration_error

    # Test error creation
    error = configuration_error("Test error")
    assert isinstance(error, ApplicationError)
    assert "Configuration error" in str(error)

    # Test result creation with error using Result pattern
    result: Result[str] = Result.failure(error)
    assert result.is_failure
    assert result.error == error
    assert result.error is not None
    assert "Configuration error" in str(result.error)
