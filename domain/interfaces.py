"""Defines the abstract interfaces for the application's core services.

These interfaces allow for decoupling the application logic from specific
implementations, facilitating testing and modularity.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from .errors import Result
from .models import (
    CleanupStats,
    PDFInfo,
    PdfPageText,
    ProcessingRequest,
    SynthesizedSegment,
    TimedAudioResult,
)

# Polled between units of work; returns True once the user has cancelled
CancellationCheck = Callable[[], bool]


def never_cancelled() -> bool:
    """Cancellation check for work that can't be cancelled."""
    return False


# --- Core Service Interfaces ---


class IFileManager(ABC):
    """Interface for managing file operations and paths."""

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
    def perform_ocr(self, image: bytes) -> Result[str]:
        """Recognize the text in an encoded image (e.g. PNG bytes)."""


class IPdfTextExtractor(ABC):
    """Interface for reading PDF documents: metadata, text and page images."""

    @abstractmethod
    def get_pdf_info(self, pdf_path: str) -> Result[PDFInfo]:
        """Page count and metadata; fails for a file that can't be read as a PDF."""

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
    def check_ready(self) -> Result[None]:
        """Report whether the engine can synthesize, without synthesizing anything.

        Checked once at startup so a missing model or runtime is reported then,
        not on the first conversion.
        """


class IAudioEncoder(ABC):
    """Interface for encoding assembled audio into a distributable format."""

    @abstractmethod
    def encode_mp3(self, wav: bytes, dest_path: str) -> Result[str]:
        """Encode WAV bytes to an MP3 at dest_path, returning that path."""


# --- Domain Engine Interfaces ---


class IDocumentEngine(ABC):
    """Interface for document processing operations using Result[T] pattern."""

    @abstractmethod
    def extract_text(self, pdf_path: str, pages: list[int] | None = None) -> Result[list[str]]:
        """Extract text from PDF with OCR fallback."""

    @abstractmethod
    def process_document(
        self,
        request: ProcessingRequest,
        audio_engine: "IAudioEngine",
        text_pipeline: "ITextPipeline",
        llm_chunk_size: int = 50000,
        is_cancelled: CancellationCheck = never_cancelled,
    ) -> Result[TimedAudioResult]:
        """Complete document processing workflow; stops early with a CANCELLED error once is_cancelled()."""


class IAudioEngine(ABC):
    """Interface for audio operations using Result[T] pattern."""

    @abstractmethod
    def generate(
        self, text_chunks: list[str], output_filename: str, is_cancelled: CancellationCheck = never_cancelled
    ) -> Result[TimedAudioResult]:
        """Synthesize text chunks into one MP3 and the timeline that read-along plays against.

        No chunk is started once is_cancelled() is true, and a cancelled run writes no file.
        """


class ITextPipeline(ABC):
    """Interface for text processing operations using Result[T] pattern."""

    @abstractmethod
    def clean_text(self, raw_text: str) -> Result[str]:
        """Clean and prepare text for TTS."""
