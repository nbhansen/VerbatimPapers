# tests/conftest.py
"""Professional test fixtures for VerbatimPapers.

Simple, reliable fixtures that real development teams use.
"""

from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import MagicMock

import pytest

from application.config.app_configs import FlaskConfig
from application.config.file_configs import FileCleanupConfig, FileConfig
from application.config.processing_configs import LLMConfig, OCRConfig, PerformanceConfig, TextProcessingConfig
from application.config.system_config import SystemConfig
from domain.config.tts_config import PiperConfig, TTSConfig, TTSEngine
from domain.errors import Result
from domain.models import (
    PageRange,
    PDFInfo,
    ProcessingRequest,
    SynthesizedSegment,
    TextSegment,
    TimedAudioResult,
    TimingMetadata,
)
from tests.test_helpers import fake_segments

# === Core Test Fixtures ===


@pytest.fixture
def temp_dir():
    """Isolated temporary directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir):
    """Standard test configuration with safe defaults."""
    return SystemConfig(
        tts=TTSConfig(engine=TTSEngine.PIPER, concurrent_requests=4, request_delay_seconds=2.0),
        files=FileConfig(
            upload_folder=str(temp_dir / "uploads"),
            audio_folder=str(temp_dir / "audio"),
            max_file_size_mb=100,
        ),
        cleanup=FileCleanupConfig(enabled=False),  # Don't interfere with tests
        text_processing=TextProcessingConfig(
            enable_cleaning=False,  # Fast tests
        ),
        performance=PerformanceConfig(),
        flask=FlaskConfig(),
        ocr=OCRConfig(),
        llm=LLMConfig(model_name="test-llm-model"),
        piper=PiperConfig(model_name="test-piper-model"),
        gemini=None,  # No external dependencies
    )


@pytest.fixture
def sample_pdf_file(temp_dir):
    """Create a fake PDF file for testing."""
    pdf_file = temp_dir / "sample.pdf"
    # Create minimal PDF-like content
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 0\ntrailer\n<< /Root 1 0 R >>\n%%EOF"
    pdf_file.write_bytes(pdf_content)
    return str(pdf_file)


# === Text Fixtures ===


@pytest.fixture
def simple_text():
    """Simple text for basic testing."""
    return "Hello world. This is a test sentence."


@pytest.fixture
def academic_text():
    """Academic text with typical research paper elements."""
    return """
    This study examined the effectiveness of automated text processing.
    Results showed a significant improvement (p < 0.001) in processing speed.
    According to Smith et al. (2020), similar approaches have been successful.
    Therefore, we recommend this method for future implementations.
    """.strip()


@pytest.fixture
def complex_text():
    """Text with challenging elements for robust testing."""
    return """
    Dr. Johnson's research at MIT (2023) found 85.7% improvement rates.
    The F(2,47) = 15.3 test results were statistically significant.
    However, further validation with larger datasets is needed.
    Key findings include: (1) faster processing, (2) better accuracy.
    """.strip()


# === Mock Service Fixtures ===


@pytest.fixture
def mock_tts_engine():
    """Professional mock TTS engine with realistic behavior."""
    mock = MagicMock()

    def synthesize_side_effect(text: str) -> Result[list[SynthesizedSegment]]:
        return Result.success(fake_segments(text))

    async def synthesize_async_side_effect(text: str) -> Result[list[SynthesizedSegment]]:
        return Result.success(fake_segments(text))

    mock.synthesize.side_effect = synthesize_side_effect
    mock.synthesize_async.side_effect = synthesize_async_side_effect

    return mock


@pytest.fixture
def mock_llm_provider():
    """Professional mock LLM provider."""
    mock = MagicMock()

    def clean_text_side_effect(text: str) -> Result[str]:
        # Simulate realistic text cleaning
        cleaned = text.replace("\n", " ").strip()
        cleaned += "... *pause*"  # Add realistic LLM enhancements
        return Result.success(cleaned)

    mock.generate_content.side_effect = clean_text_side_effect
    mock.process_text.side_effect = clean_text_side_effect

    return mock


@pytest.fixture
def mock_file_manager(temp_dir):
    """Professional mock file manager with real file operations."""
    mock = MagicMock()

    # Track files for cleanup
    created_files = []

    def save_output_file(content: bytes, filename: str) -> str:
        filepath = temp_dir / filename
        filepath.write_bytes(content)
        created_files.append(str(filepath))
        return str(filepath)

    def save_temp_file(content: bytes, suffix: str = ".tmp") -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as f:
            f.write(content)
            created_files.append(f.name)
            return f.name

    mock.save_output_file.side_effect = save_output_file
    mock.save_temp_file.side_effect = save_temp_file
    mock.get_output_dir.return_value = str(temp_dir)
    mock.created_files = created_files

    return mock


@pytest.fixture
def mock_ocr_provider():
    """Professional mock OCR provider."""
    mock = MagicMock()

    mock.extract_text.return_value = "This is extracted text from the PDF document."
    mock.get_pdf_info.return_value = PDFInfo(total_pages=3, title="Sample Document", author="Test Author")

    return mock


# === Domain Model Fixtures ===


@pytest.fixture
def sample_processing_request(sample_pdf_file):
    """Standard processing request for testing."""
    return ProcessingRequest(pdf_path=sample_pdf_file, output_name="test_output", page_range=PageRange())


@pytest.fixture
def sample_text_segments():
    """Sample text segments with realistic timing."""
    return [
        TextSegment(
            text="First sentence of the document.",
            start_time=0.0,
            duration=2.1,
            segment_type="sentence",
            chunk_index=0,
            sentence_index=0,
        ),
        TextSegment(
            text="Second sentence with more content to process.",
            start_time=2.1,
            duration=3.2,
            segment_type="sentence",
            chunk_index=0,
            sentence_index=1,
        ),
        TextSegment(
            text="Final sentence to complete the test.",
            start_time=5.3,
            duration=2.8,
            segment_type="sentence",
            chunk_index=0,
            sentence_index=2,
        ),
    ]


@pytest.fixture
def sample_timing_metadata(sample_text_segments):
    """Sample timing metadata for audio testing."""
    return TimingMetadata(
        total_duration=8.1, text_segments=sample_text_segments, audio_files=["test_output_combined.wav"]
    )


@pytest.fixture
def sample_timed_audio_result(sample_timing_metadata):
    """Complete timed audio result for integration testing."""
    return TimedAudioResult(
        audio_files=["test_output_combined.wav"],
        combined_mp3="test_output_combined.wav",
        timing_data=sample_timing_metadata,
    )


# === Professional Test Utilities ===


@pytest.fixture
def assert_audio_quality():
    """Utility for asserting audio output quality."""

    def _assert_audio_quality(audio_data: bytes, min_size: int = 100) -> None:
        assert audio_data is not None, "Audio data should not be None"
        assert len(audio_data) >= min_size, f"Audio data too small: {len(audio_data)} bytes"
        assert audio_data.startswith(b"fake_audio_") or b"RIFF" in audio_data[:20], "Invalid audio format"

    return _assert_audio_quality


@pytest.fixture
def assert_timing_accuracy():
    """Utility for asserting timing accuracy."""

    def _assert_timing_accuracy(segments: list[Any], tolerance: float = 0.1) -> None:
        for i, segment in enumerate(segments):
            assert segment.duration > 0, f"Segment {i} has invalid duration: {segment.duration}"
            assert segment.start_time >= 0, f"Segment {i} has negative start time: {segment.start_time}"

            if i > 0:
                prev_end = segments[i - 1].start_time + segments[i - 1].duration
                current_start = segment.start_time
                gap = abs(current_start - prev_end)
                assert gap <= tolerance, f"Timing gap too large between segments {i - 1} and {i}: {gap}s"

    return _assert_timing_accuracy


# === pytest Configuration ===


def pytest_configure(config):
    """Configure pytest with professional markers."""
    config.addinivalue_line("markers", "unit: Fast unit tests with no external dependencies")
    config.addinivalue_line("markers", "integration: Integration tests with mocked external services")
    config.addinivalue_line("markers", "external: Tests requiring real external services (manual)")
    config.addinivalue_line("markers", "slow: Tests taking more than 5 seconds")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on location."""
    for item in items:
        # Auto-mark tests based on directory
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark external dependency tests
        if hasattr(item, "fixturenames") and any(
            "real_" in fixture or "external_" in fixture for fixture in item.fixturenames
        ):
            item.add_marker(pytest.mark.external)


# === Cleanup ===


@pytest.fixture(autouse=True)
def cleanup_test_files(temp_dir):
    """Automatically clean up test files after each test."""
    return
    # Cleanup happens automatically with temp_dir fixture


# === Skip-If-Unavailable Fixtures for Real External Services ===


def _check_command_available(command: str) -> bool:
    """Check if a command is available on the system."""
    import shutil

    return shutil.which(command) is not None


def _check_piper_available() -> bool:
    """Check if Piper TTS is available (command line or Python library)."""
    # Check command line
    if _check_command_available("piper"):
        return True
    # Check Python library
    try:
        from piper.voice import PiperVoice  # noqa: F401

        return True
    except ImportError:
        pass
    # Check project local piper binary
    local_piper = Path.cwd() / "piper"
    return local_piper.exists() and local_piper.is_file()


@pytest.fixture
def requires_tesseract():
    """Skip test if Tesseract OCR is not available."""
    if not _check_command_available("tesseract"):
        pytest.skip("Tesseract OCR not available")


@pytest.fixture
def requires_piper():
    """Skip test if Piper TTS is not available."""
    if not _check_piper_available():
        pytest.skip("Piper TTS not available")


@pytest.fixture
def requires_ffmpeg():
    """Skip test if FFmpeg is not available."""
    if not _check_command_available("ffmpeg"):
        pytest.skip("FFmpeg not available")


@pytest.fixture
def requires_poppler():
    """Skip test if Poppler (pdftoppm/pdfinfo) is not available."""
    if not _check_command_available("pdftoppm"):
        pytest.skip("Poppler utilities not available")


@pytest.fixture
def real_piper_tts(requires_piper, temp_dir):
    """Create a real Piper TTS provider for integration testing."""
    from domain.config import PiperConfig
    from infrastructure.tts.piper_tts_provider import PiperTTSProvider

    config = PiperConfig(
        model_name="en_US-lessac-medium",
        download_dir=str(temp_dir / "piper_models"),
    )
    provider = PiperTTSProvider(config, project_root=str(Path.cwd()))
    return provider


@pytest.fixture
def real_tesseract_ocr(requires_tesseract, requires_poppler):
    """Create a real Tesseract OCR provider for integration testing."""
    from infrastructure.ocr.tesseract_ocr_provider import TesseractOCRProvider

    return TesseractOCRProvider()


@pytest.fixture
def real_file_manager(temp_dir):
    """Create a real FileManager for integration testing."""
    from infrastructure.file.file_manager import FileManager

    upload_dir = temp_dir / "uploads"
    output_dir = temp_dir / "output"
    upload_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return FileManager(str(upload_dir), str(output_dir))
