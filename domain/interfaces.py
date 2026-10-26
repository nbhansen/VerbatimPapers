"""Defines the abstract interfaces for the application's core services.

These interfaces allow for decoupling the application logic from specific
implementations, facilitating testing and modularity.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from .errors import Result
from .models import (
    CleanupStats,
    PageRange,
    PDFInfo,
    PdfPageText,
    ProcessingRequest,
    SynthesizedSegment,
    TimedAudioResult,
)

# --- Core Service Interfaces ---


class IFileManager(ABC):
    """Interface for managing file operations and paths."""

    @abstractmethod
    def save_temp_file(self, content: bytes, suffix: str = ".tmp") -> str:
        """Saves content to a temporary file and returns its path."""

    @abstractmethod
    def save_output_file(self, content: bytes, filename: str) -> str:
        """Saves content to a final output file and returns its path."""

    @abstractmethod
    def delete_file(self, filepath: str) -> None:
        """Deletes a file at the given path."""

    @abstractmethod
    def get_output_dir(self) -> str:
        """Returns the path to the output directory."""

    @abstractmethod
    def cleanup_old_files(self, max_age_hours: float) -> Result[CleanupStats]:
        """Deletes output and upload files older than max_age_hours; fails on non-positive or non-finite age."""


class ILLMProvider(ABC):
    """Interface for a Large Language Model provider."""

    @abstractmethod
    def generate_content(self, prompt: str) -> Result[str]:
        """Generates content based on a prompt."""


class IOCRProvider(ABC):
    """Interface for an Optical Character Recognition provider."""

    @abstractmethod
    def perform_ocr(self, image_path: str) -> Result[str]:
        """Performs OCR on an image and returns the extracted text."""

    @abstractmethod
    def get_pdf_info(self, pdf_path: str) -> PDFInfo:
        """Get PDF document information including page count and metadata."""

    @abstractmethod
    def validate_range(self, pdf_path: str, page_range: PageRange) -> dict[str, Any]:
        """Validate page range against PDF document."""


class IPdfTextExtractor(ABC):
    """Interface for extracting text and page images from PDF documents."""

    @abstractmethod
    def extract_text_by_page(self, pdf_path: str, pages: list[int] | None = None) -> Result[list[PdfPageText]]:
        """Extract raw text for the requested 0-based page indices (all pages if None).

        Out-of-range indices are skipped; a page whose extraction fails yields
        empty text so callers can decide on a fallback (e.g. OCR).
        """

    @abstractmethod
    def render_page_image(self, pdf_path: str, page_index: int, resolution: int = 300) -> Result[bytes]:
        """Render a single 0-based page as PNG bytes (for OCR fallback)."""


class ITTSEngine(ABC):
    """Interface for a Text-to-Speech engine.

    Engines return segments, not a finished audio file. Engines that synthesize
    sentence by sentence (Piper) return one segment per sentence, so the text and
    its audio stay associated and timing can be measured rather than estimated.
    Engines with no internal structure (Gemini) return a single segment.

    Turning segments into a playable file - inter-sentence silence, level
    handling, container format - is AudioAssembler's job, not an engine's.
    """

    @abstractmethod
    def synthesize(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Synthesize text into one or more segments.

        Returns:
            Result[Sequence[SynthesizedSegment]]: Segments in playback order, or failure.
        """

    @abstractmethod
    async def synthesize_async(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Synthesize text asynchronously.

        Engines without native async support may wrap the sync method.
        """

    @abstractmethod
    def supports_ssml(self) -> bool:
        """Check if this TTS engine supports SSML markup."""


# --- Domain Engine Interfaces ---


class IDocumentEngine(ABC):
    """Interface for document processing operations using Result[T] pattern."""

    @abstractmethod
    def get_pdf_info(self, pdf_path: str) -> Result[PDFInfo]:
        """Get PDF metadata and information."""

    @abstractmethod
    def validate_page_range(self, pdf_path: str, page_range: PageRange) -> Result[dict[str, Any]]:
        """Validate requested page range."""

    @abstractmethod
    def extract_text(self, pdf_path: str, pages: list[int] | None = None) -> Result[list[str]]:
        """Extract text from PDF with OCR fallback."""

    @abstractmethod
    def process_document(
        self,
        request: ProcessingRequest,
        audio_engine: "IAudioEngine",
        text_pipeline: "ITextPipeline",
        enable_timing: bool = False,
        llm_chunk_size: int = 50000,
    ) -> Result[TimedAudioResult]:
        """Complete document processing workflow."""


class IAudioEngine(ABC):
    """Interface for audio operations using Result[T] pattern."""

    @abstractmethod
    def generate_with_timing(self, text_chunks: list[str], output_filename: str) -> Result[TimedAudioResult]:
        """Generate audio with timing data from text chunks."""

    @abstractmethod
    def generate_simple_audio(self, text_chunks: list[str], output_filename: str) -> Result[TimedAudioResult]:
        """Generate audio without timing complexity - for regular uploads."""


class ITextPipeline(ABC):
    """Interface for text processing operations using Result[T] pattern."""

    @abstractmethod
    def clean_text(self, raw_text: str) -> Result[str]:
        """Clean and prepare text for TTS."""

    @abstractmethod
    def split_into_sentences(self, text: str) -> Result[list[str]]:
        """Split text into sentences for processing."""
