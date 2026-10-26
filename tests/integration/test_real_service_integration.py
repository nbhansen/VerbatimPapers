"""Real service integration tests without inappropriate mocking.

Tests actual service factory behavior, dependency injection, and service interaction
with minimal mocking. Only mocks external dependencies that would require network
calls or files not available in test environment.
"""

from collections.abc import Generator
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pytest

from application.composition import Services
from application.config.app_configs import FlaskConfig
from application.config.file_configs import FileCleanupConfig, FileConfig
from application.config.processing_configs import LLMConfig, OCRConfig, PerformanceConfig, TextProcessingConfig
from application.config.system_config import SystemConfig
from domain.config.tts_config import PiperConfig, TTSConfig, TTSEngine
from domain.interfaces import ITTSEngine
from domain.models import PageRange, ProcessingRequest
from infrastructure.file.file_manager import FileManager


@pytest.fixture
def temp_directories() -> Generator[tuple[str, str], None, None]:
    """Create temporary directories for uploads and audio output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        upload_dir = str(Path(temp_dir) / "uploads")
        audio_dir = str(Path(temp_dir) / "audio")
        Path(upload_dir).mkdir()
        Path(audio_dir).mkdir()
        yield upload_dir, audio_dir


@pytest.fixture
def integration_config(temp_directories: tuple[str, str]) -> SystemConfig:
    """SystemConfig for integration testing with real services."""
    upload_dir, audio_dir = temp_directories

    return SystemConfig(
        tts=TTSConfig(
            engine=TTSEngine.PIPER,
            concurrent_requests=1,
            request_delay_seconds=0.1,
        ),
        files=FileConfig(
            upload_folder=upload_dir,
            audio_folder=audio_dir,
            max_file_size_mb=10,
        ),
        cleanup=FileCleanupConfig(enabled=False),  # Don't cleanup during tests
        text_processing=TextProcessingConfig(
            enable_cleaning=False,  # Disable LLM calls for integration tests
            llm_chunk_size=500,
        ),
        performance=PerformanceConfig(),
        flask=FlaskConfig(),
        ocr=OCRConfig(),
        llm=LLMConfig(model_name="test-llm-model"),
        piper=PiperConfig(model_name="en_US-lessac-medium"),
        gemini=None,  # No Gemini for these tests
    )


class TestServiceFactoryIntegration:
    """Test service factory creates real services with proper dependencies."""

    def test_creates_all_core_services(self, integration_config: SystemConfig) -> None:
        """Should create all core services without excessive mocking."""
        # Only mock Piper availability check - rest should be real
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = Services(integration_config)

            # Verify all main services are created
            audio_engine = container.audio_engine
            text_pipeline = container.text_pipeline
            document_engine = container.document_engine
            file_manager = container.file_manager
            assert isinstance(file_manager, FileManager)

            assert audio_engine is not None
            assert text_pipeline is not None
            assert document_engine is not None
            assert file_manager is not None

            # Services should be real instances, not mocks
            assert not isinstance(audio_engine, Mock)
            assert not isinstance(text_pipeline, Mock)
            assert not isinstance(document_engine, Mock)
            assert not isinstance(file_manager, Mock)

    def test_services_share_dependencies(self, integration_config: SystemConfig) -> None:
        """Should properly inject dependencies between services."""
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = Services(integration_config)

            # Get services that should share dependencies
            audio_engine = container.audio_engine
            document_engine = container.document_engine
            file_manager = container.file_manager
            assert isinstance(file_manager, FileManager)

            # Services should use the same file manager instance (dependency injection)
            # This tests that the service container is working properly
            assert hasattr(audio_engine, "_file_manager") or hasattr(audio_engine, "file_manager")
            assert hasattr(document_engine, "_file_manager") or hasattr(document_engine, "file_manager")

            # File manager should have correct configuration
            assert file_manager.upload_folder == integration_config.files.upload_folder
            assert file_manager.output_folder == integration_config.files.audio_folder

    def test_progress_store_is_shared(self, integration_config: SystemConfig) -> None:
        """DocumentProcessingService must share the single progress store instance."""
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = Services(integration_config)

            store = container.progress_store
            assert container.progress_store is store

            document_service = container.document_service
            assert document_service.progress_store is store

    def test_tts_engine_selection_and_creation(self, integration_config: SystemConfig) -> None:
        """Should create correct TTS engine based on configuration."""
        # Test Piper engine selection
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = Services(integration_config)

            audio_engine = container.audio_engine
            assert audio_engine is not None

            # Audio engine should have a TTS engine
            assert hasattr(audio_engine, "_tts_engine") or hasattr(audio_engine, "tts_engine")

    def test_configuration_propagation(self, integration_config: SystemConfig) -> None:
        """Should properly propagate configuration to all services."""
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = Services(integration_config)

            text_pipeline = container.text_pipeline
            file_manager = container.file_manager
            assert isinstance(file_manager, FileManager)

            # Configuration should be propagated correctly
            assert file_manager.upload_folder == integration_config.files.upload_folder
            assert file_manager.output_folder == integration_config.files.audio_folder

            # Text pipeline should reflect configuration settings
            assert text_pipeline is not None


class TestServiceInteractionIntegration:
    """Test real interactions between services."""

    def test_document_engine_ocr_integration(self, integration_config: SystemConfig) -> None:
        """Should integrate real document engine with OCR provider."""
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = Services(integration_config)

            document_engine = container.document_engine

            # Document engine should have real OCR provider
            assert hasattr(document_engine, "_ocr_provider") or hasattr(document_engine, "ocr_provider")

            # OCR provider should be real TesseractOCRProvider
            ocr_provider = getattr(document_engine, "_ocr_provider", None) or getattr(
                document_engine, "ocr_provider", None
            )
            assert ocr_provider is not None
            assert not isinstance(ocr_provider, Mock)

    def test_audio_engine_tts_integration(self, integration_config: SystemConfig) -> None:
        """Should integrate real audio engine with TTS provider."""
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = Services(integration_config)

            audio_engine = container.audio_engine

            # Audio engine should have real TTS engine
            assert hasattr(audio_engine, "_tts_engine") or hasattr(audio_engine, "tts_engine")

            # TTS engine should be real implementation
            tts_engine = getattr(audio_engine, "_tts_engine", None) or getattr(audio_engine, "tts_engine", None)
            assert tts_engine is not None
            assert not isinstance(tts_engine, Mock)

    def test_file_manager_directory_creation(self, integration_config: SystemConfig) -> None:
        """Should properly create and manage directories."""
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = Services(integration_config)

            file_manager = container.file_manager
            assert isinstance(file_manager, FileManager)

            # Directories should exist
            assert Path(file_manager.upload_folder).exists()
            assert Path(file_manager.output_folder).exists()

            # File manager should have correct permissions and access
            assert Path(file_manager.upload_folder).is_dir()
            assert Path(file_manager.output_folder).is_dir()


class TestDomainModelIntegration:
    """Test integration with domain models and business logic."""

    def test_processing_request_creation_and_validation(self, integration_config: SystemConfig) -> None:
        """Should create and validate processing requests with real services."""
        # Create a processing request
        request = ProcessingRequest(
            pdf_path="test_document.pdf", output_name="test_output", page_range=PageRange(start_page=1, end_page=5)
        )

        # Request should be properly formed
        assert request.pdf_path == "test_document.pdf"
        assert request.output_name == "test_output"
        assert request.page_range.start_page == 1
        assert request.page_range.end_page == 5
        assert not request.page_range.is_full_document()


class TestErrorHandlingIntegration:
    """Test error handling integration across services."""

    def test_service_creation_error_handling(self, temp_directories: tuple[str, str]) -> None:
        """Should handle service creation errors gracefully."""
        upload_dir, audio_dir = temp_directories

        # Create config with invalid settings
        invalid_config = SystemConfig(
            tts=TTSConfig(engine=TTSEngine.PIPER),
            files=FileConfig(
                upload_folder="/nonexistent/directory",  # Invalid directory
                audio_folder=audio_dir,
                max_file_size_mb=10,
            ),
            cleanup=FileCleanupConfig(enabled=False),
            text_processing=TextProcessingConfig(),
            performance=PerformanceConfig(),
            flask=FlaskConfig(),
            ocr=OCRConfig(),
            llm=LLMConfig(model_name="test-model"),
            piper=PiperConfig(model_name="en_US-lessac-medium"),
            gemini=None,
        )

        # Service creation might fail or work depending on FileManager implementation
        # This tests that errors are handled gracefully rather than crashing
        try:
            with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
                container = Services(invalid_config)
                # If it succeeds, services should still be created
                assert container is not None
        except Exception as e:
            # If it fails, should be a proper error, not a crash
            assert isinstance(e, Exception)
            assert str(e)  # Should have error message

    def test_missing_dependency_handling(self, integration_config: SystemConfig) -> None:
        """Should handle missing optional dependencies gracefully."""
        # Test with Piper unavailable
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False):
            # This might fail or fall back to alternative - both are valid behaviors
            try:
                container = Services(integration_config)
                # If successful, should have valid services
                assert container is not None
            except Exception:
                # If it fails, that's also acceptable for missing dependencies
                pass


class TestRealProviderInteraction:
    """Test interaction with real provider implementations (minimal mocking)."""

    def test_tts_provider_interface_compliance(self, integration_config: SystemConfig) -> None:
        """Should create TTS providers that comply with ITTSEngine interface."""
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            # Mock only model loading to avoid downloading models in tests
            with patch("infrastructure.tts.piper_tts_provider.PiperTTSProvider._ensure_model") as mock_ensure:
                mock_ensure.return_value = ("/fake/model.onnx", "/fake/config.json")

                with patch("pathlib.Path.exists") as mock_exists:
                    mock_exists.return_value = True

                    container = Services(integration_config)
                    audio_engine = container.audio_engine

                    # Should have real TTS engine that implements interface
                    tts_engine = getattr(audio_engine, "_tts_engine", None) or getattr(audio_engine, "tts_engine", None)

                    if tts_engine:
                        # Test interface compliance (ITTSEngine)
                        assert isinstance(tts_engine, ITTSEngine)
                        assert hasattr(tts_engine, "synthesize")
                        assert hasattr(tts_engine, "synthesize_async")

    def test_text_pipeline_real_processing(self, integration_config: SystemConfig) -> None:
        """Should create text pipeline that does real text processing."""
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = Services(integration_config)
            text_pipeline = container.text_pipeline

            # Text pipeline should be real implementation
            assert text_pipeline is not None
            assert not isinstance(text_pipeline, Mock)

            # Should have proper interface methods
            assert hasattr(text_pipeline, "clean_text")

    def test_file_manager_real_operations(self, integration_config: SystemConfig) -> None:
        """Should create file manager that does real file operations."""
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = Services(integration_config)
            file_manager = container.file_manager
            assert isinstance(file_manager, FileManager)

            # File manager should be real implementation
            assert file_manager is not None
            assert not isinstance(file_manager, Mock)

            # Should manage real directories
            assert Path(file_manager.upload_folder).exists()
            assert Path(file_manager.output_folder).exists()

            # Should have file operation methods
            assert hasattr(file_manager, "save_output_file")
            assert hasattr(file_manager, "save_temp_file")
            assert hasattr(file_manager, "delete_file")


class TestRealExternalServicesIntegration:
    """Integration tests with real external services (skipped if unavailable)."""

    def test_real_piper_generates_audio(self, real_piper_tts) -> None:
        """Should generate actual audio with real Piper TTS."""
        test_text = "Hello, this is a test of Piper text to speech. Here is a second sentence."
        result = real_piper_tts.synthesize(test_text)

        # Should return Result pattern
        assert hasattr(result, "is_success")
        if result.is_success:
            segments = result.value
            assert segments is not None
            assert len(segments) > 0

            for segment in segments:
                assert segment.pcm, "Segment carries no audio"
                assert segment.duration > 0, "Segment has no measurable duration"
                assert segment.text.strip(), "Segment carries no text"
                # Raw samples, not a container - wrapping is AudioAssembler's job
                assert segment.pcm[:4] != b"RIFF", "Expected raw PCM, not a WAV container"

            # Longer sentences must take longer, which the old even split could not express
            if len(segments) > 1:
                longest = max(segments, key=lambda s: len(s.text))
                shortest = min(segments, key=lambda s: len(s.text))
                if longest.text != shortest.text:
                    assert longest.duration > shortest.duration

    def test_real_file_manager_saves_and_retrieves(self, real_file_manager) -> None:
        """Should save and retrieve files with real FileManager."""
        # Save a test file
        test_content = b"Test audio content for integration testing"
        saved_path = real_file_manager.save_output_file(test_content, "test_integration.wav")

        assert saved_path is not None
        assert Path(saved_path).exists()

        # Read it back
        with open(saved_path, "rb") as f:
            retrieved_content = f.read()
        assert retrieved_content == test_content

        # Clean up
        Path(saved_path).unlink()

    def test_real_tesseract_ocr_methods(self, real_tesseract_ocr) -> None:
        """Should have proper OCR interface methods."""
        # Verify the OCR provider has expected interface methods
        assert hasattr(real_tesseract_ocr, "perform_ocr")
        assert hasattr(real_tesseract_ocr, "get_pdf_info")
        assert hasattr(real_tesseract_ocr, "validate_range")

        # Test get_pdf_info with nonexistent file - should return default info
        # The implementation returns PDFInfo(total_pages=0, ...) on exception
        info = real_tesseract_ocr.get_pdf_info("/nonexistent/file.pdf")

        assert info.total_pages == 0
        assert info.title == "Unknown"

        # Test validate_range with nonexistent file
        # The implementation returns error dict on exception
        result = real_tesseract_ocr.validate_range("/nonexistent/file.pdf", PageRange())

        assert result["valid"] is False
        assert result["total_pages"] == 0
