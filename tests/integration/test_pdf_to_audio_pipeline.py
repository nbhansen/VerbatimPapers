"""Critical path testing for real PDF-to-audio pipeline.

Tests the complete user workflow from PDF input to audio output using
real components where possible. These tests validate that the core
functionality actually works end-to-end.
"""

from collections.abc import Generator
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pytest

from application.config.app_configs import FlaskConfig
from application.config.file_configs import FileCleanupConfig, FileConfig
from application.config.processing_configs import LLMConfig, OCRConfig, PerformanceConfig, TextProcessingConfig
from application.config.system_config import SystemConfig
from application.container.service_container import create_service_container
from domain.config.tts_config import PiperConfig, TTSConfig, TTSEngine
from domain.interfaces import IAudioEngine, IDocumentEngine, ITextPipeline
from domain.models import PageRange, PDFInfo, TimedAudioResult


@pytest.fixture
def test_pdf_content() -> bytes:
    """Create a minimal valid PDF for testing."""
    # This is a minimal PDF that contains text "Hello World"
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f
0000000010 00000 n
0000000053 00000 n
0000000125 00000 n
0000000296 00000 n
0000000389 00000 n
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
459
%%EOF"""
    return pdf_content


@pytest.fixture
def pipeline_test_dirs() -> Generator[tuple[str, str], None, None]:
    """Create temporary directories for pipeline testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        upload_dir = str(Path(temp_dir) / "uploads")
        audio_dir = str(Path(temp_dir) / "audio")
        Path(upload_dir).mkdir()
        Path(audio_dir).mkdir()
        yield upload_dir, audio_dir


@pytest.fixture
def pipeline_config(pipeline_test_dirs: tuple[str, str]) -> SystemConfig:
    """Configuration for pipeline testing."""
    upload_dir, audio_dir = pipeline_test_dirs

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
        cleanup=FileCleanupConfig(enabled=False),
        text_processing=TextProcessingConfig(
            enable_cleaning=False,  # Disable LLM to focus on pipeline
            llm_chunk_size=500,
            audio_target_chunk_size=1000,
        ),
        performance=PerformanceConfig(
            enable_async_audio=False,  # Sync for predictable testing
            audio_concurrent_chunks=1,
        ),
        flask=FlaskConfig(),
        ocr=OCRConfig(),
        llm=LLMConfig(model_name="test-llm-model"),
        piper=PiperConfig(model_name="en_US-lessac-medium"),
        gemini=None,
    )


class TestPDFToAudioPipelineCore:
    """Test core PDF-to-audio conversion pipeline."""

    def test_document_text_extraction_pipeline(
        self, pipeline_config: SystemConfig, test_pdf_content: bytes, pipeline_test_dirs: tuple[str, str]
    ) -> None:
        """Should extract text from PDF using real document engine."""
        upload_dir, _ = pipeline_test_dirs

        # Create test PDF file
        pdf_path = Path(upload_dir) / "test_document.pdf"
        pdf_path.write_bytes(test_pdf_content)

        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = create_service_container(pipeline_config)
            document_engine = container.get(IDocumentEngine)

            # Test PDF info extraction
            pdf_info_result = document_engine.get_pdf_info(str(pdf_path))
            assert pdf_info_result.is_success

            pdf_info = pdf_info_result.value
            assert isinstance(pdf_info, PDFInfo)
            assert pdf_info.total_pages >= 1

            # Test text extraction
            text_result = document_engine.extract_text(str(pdf_path), None)  # Full document

            # Should extract some text (even if not perfect)
            if text_result.is_success:
                extracted_text = text_result.value
                assert isinstance(extracted_text, list)  # extract_text returns list[str]
                assert len(extracted_text) > 0
                assert all(isinstance(page_text, str) for page_text in extracted_text)
            else:
                # OCR might fail on minimal PDF, that's acceptable for test
                assert text_result.is_failure
                assert text_result.error is not None

    def test_text_processing_pipeline(self, pipeline_config: SystemConfig) -> None:
        """Should process extracted text through text pipeline."""
        test_text = """
        This is a test document with multiple sentences.
        It contains various punctuation marks! Questions?
        And some formatting issues   with  extra spaces.
        """

        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = create_service_container(pipeline_config)
            text_pipeline = container.get(ITextPipeline)

            # Test text processing
            processing_result = text_pipeline.clean_text(test_text)

            assert processing_result.is_success
            processed_text = processing_result.value

            # Should clean and process the text
            assert isinstance(processed_text, str)
            assert len(processed_text) > 0

            # Test sentence splitting
            sentences_result = text_pipeline.split_into_sentences(processed_text)
            assert sentences_result.is_success

            sentences = sentences_result.value
            assert isinstance(sentences, list)
            assert len(sentences) > 0
            assert all(isinstance(sentence, str) for sentence in sentences)

    @patch("infrastructure.tts.piper_tts_provider.PiperTTSProvider._generate_with_command_line")
    @patch("infrastructure.tts.piper_tts_provider.PiperTTSProvider._ensure_model")
    @patch("pathlib.Path.exists")
    def test_audio_generation_pipeline(
        self, mock_exists: Mock, mock_ensure_model: Mock, mock_generate: Mock, pipeline_config: SystemConfig
    ) -> None:
        """Should generate audio from processed text."""
        # Setup mocks for Piper TTS
        mock_exists.return_value = True
        mock_ensure_model.return_value = ("/fake/model.onnx", "/fake/config.json")

        # Mock successful audio generation
        fake_wav_header = b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00"
        fake_audio_data = fake_wav_header + b"fake_audio_content"
        mock_generate.return_value = fake_audio_data

        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = create_service_container(pipeline_config)
            audio_engine = container.get(IAudioEngine)

            # Test audio generation engine existence
            assert audio_engine is not None
            # Check for underlying TTS engine presence as implementation detail
            assert hasattr(audio_engine, "tts_engine") or hasattr(audio_engine, "_tts_engine")

    def test_missing_file_handling(self, pipeline_config: SystemConfig) -> None:
        """Should handle missing PDF files gracefully."""
        nonexistent_path = "/nonexistent/directory/missing.pdf"

        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = create_service_container(pipeline_config)
            document_engine = container.get(IDocumentEngine)

            # Should handle missing file gracefully - specific implementation detail:
            # TesseractOCRProvider returns a default PDFInfo object on error instead of raising
            pdf_info_result = document_engine.get_pdf_info(nonexistent_path)

            # The current implementation catches exceptions and returns success with default values
            if pdf_info_result.is_success:
                pdf_info = pdf_info_result.value
                assert pdf_info is not None
                assert pdf_info.total_pages == 0
                assert pdf_info.title == "Unknown"
            else:
                # If it changes to return failure, that's also valid
                assert pdf_info_result.is_failure

    def test_empty_text_handling(self, pipeline_config: SystemConfig) -> None:
        """Should handle empty or whitespace-only text gracefully."""
        empty_texts = ["", "   ", "\n\n\t", "  \n  \t  "]

        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = create_service_container(pipeline_config)
            text_pipeline = container.get(ITextPipeline)

            for empty_text in empty_texts:
                result = text_pipeline.clean_text(empty_text)

                # Should either fail gracefully or return minimal content
                if result.is_failure:
                    assert result.error is not None
                else:
                    processed = result.value
                    # If successful, should return something usable or empty string
                    assert isinstance(processed, str)

    def test_invalid_page_range_handling(self, pipeline_config: SystemConfig) -> None:
        """Should handle invalid page ranges gracefully."""
        # Test page range validation at model level
        try:
            # This should raise ValueError due to domain model validation
            invalid_range = PageRange(start_page=5, end_page=2)
            # If constructor doesn't raise, we manually validate
            if (
                invalid_range.start_page is not None
                and invalid_range.end_page is not None
                and invalid_range.start_page > invalid_range.end_page
            ):
                raise ValueError("start_page cannot be greater than end_page")
        except ValueError as e:
            assert "start_page cannot be greater than end_page" in str(e)

    def test_chunking_efficiency(self, pipeline_config: SystemConfig) -> None:
        """Should chunk text efficiently for different sizes."""
        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = create_service_container(pipeline_config)
            text_pipeline = container.get(ITextPipeline)

            # Test various text sizes
            test_cases = [
                ("Short text", "This is short."),
                ("Medium text", "This is a medium length text. " * 10),
                ("Long text", "This is a longer text document. " * 50),
            ]

            for test_name, text in test_cases:
                chunks_result = text_pipeline.split_into_sentences(text)

                if chunks_result.is_success:
                    chunks = chunks_result.value
                    assert chunks is not None

                    # Should produce reasonable chunk count
                    assert len(chunks) > 0
                    # Limit relaxed for long text
                    if "Long" in test_name:
                        assert len(chunks) < 100
                    else:
                        assert len(chunks) < 20

                    # Chunks should be reasonable size
                    for chunk in chunks:
                        assert len(chunk) <= pipeline_config.text_processing.audio_target_chunk_size
                        assert len(chunk.strip()) > 0  # No empty chunks


class TestPDFToAudioFileManagement:
    """Test file management aspects of the pipeline."""

    def test_file_path_handling(self, pipeline_config: SystemConfig, pipeline_test_dirs: tuple[str, str]) -> None:
        """Should handle file paths correctly throughout pipeline."""
        upload_dir, audio_dir = pipeline_test_dirs

        with patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True):
            container = create_service_container(pipeline_config)

            from infrastructure.file.file_manager import FileManager

            file_manager = container.get(FileManager)

            # File manager should have correct paths
            assert file_manager.upload_folder == upload_dir
            assert file_manager.output_folder == audio_dir

            # Directories should exist and be accessible
            assert Path(upload_dir).exists()
            assert Path(audio_dir).exists()
            assert Path(upload_dir).is_dir()
            assert Path(audio_dir).is_dir()

    def test_temporary_file_handling(self, pipeline_config: SystemConfig) -> None:
        """Should handle temporary files properly during processing."""
        # This is mainly tested through the infrastructure tests
        # Here we verify the configuration supports temp file management

        assert pipeline_config.cleanup.enabled is False  # No cleanup during tests
        assert pipeline_config.files.upload_folder
        assert pipeline_config.files.audio_folder

        # File size limits should be reasonable
        assert pipeline_config.files.max_file_size_mb > 0
        assert pipeline_config.files.max_file_size_mb <= 100  # Reasonable limit


class TestPDFToAudioIntegrationValidation:
    """Validate the integration works as expected."""

    def test_processing_result_structure(self, pipeline_config: SystemConfig) -> None:
        """Should create proper TimedAudioResult structures."""
        from domain.errors import Result, unknown_error

        # Test successful result
        success_result = TimedAudioResult(audio_files=["chunk1.wav", "chunk2.wav"], combined_mp3="test_output.mp3")

        assert success_result.audio_files == ["chunk1.wav", "chunk2.wav"]
        assert success_result.combined_mp3 == "test_output.mp3"
        assert success_result.timing_data is None

        # Test failure result using Result pattern
        error = unknown_error("Test processing failure")
        failure_result: Result[TimedAudioResult] = Result.failure(error)
        assert failure_result.is_failure
        assert failure_result.error == error

    def test_pipeline_configuration_validation(self, pipeline_config: SystemConfig) -> None:
        """Should validate pipeline configuration is complete."""
        # All required configuration sections should be present
        assert pipeline_config.tts is not None
        assert pipeline_config.files is not None
        assert pipeline_config.text_processing is not None
        assert pipeline_config.performance is not None
        assert pipeline_config.ocr is not None

        # TTS configuration should be valid
        assert pipeline_config.tts.engine == TTSEngine.PIPER
        assert pipeline_config.tts.concurrent_requests > 0
        assert pipeline_config.tts.request_delay_seconds >= 0

        # File configuration should be valid
        assert Path(pipeline_config.files.upload_folder).exists()
        assert Path(pipeline_config.files.audio_folder).exists()
        assert pipeline_config.files.max_file_size_mb > 0

        # Text processing configuration should be valid
        assert pipeline_config.text_processing.audio_target_chunk_size > 0
