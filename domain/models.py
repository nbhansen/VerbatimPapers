# domain/models.py
"""Domain models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageRange:
    """Range specification for document page processing."""

    start_page: int | None = None
    end_page: int | None = None

    def is_full_document(self) -> bool:
        """Return True if this range covers the entire document."""
        return self.start_page is None and self.end_page is None


@dataclass(frozen=True)
class ProcessingRequest:
    """Request specification for PDF processing operations."""

    pdf_path: str
    output_name: str
    page_range: PageRange


@dataclass(frozen=True)
class PDFInfo:
    """Information about a PDF document."""

    total_pages: int
    title: str
    author: str


@dataclass(frozen=True)
class PdfPageText:
    """Raw text extracted from a single PDF page."""

    page_index: int  # 0-based
    text: str


@dataclass(frozen=True)
class CleanupStats:
    """Outcome of an age-based file cleanup pass."""

    files_removed: int
    bytes_freed: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SynthesizedSegment:
    """One stretch of speech and the text that produced it.

    This is what TTS engines return. Keeping the text/audio correspondence intact
    is the point: durations here are measured, not estimated, so read-along timing
    does not have to guess at how a blob of audio divides between its sentences.

    `pcm` is raw samples with no container - assembling those into a playable file,
    and deciding what silence goes between them, belongs to AudioAssembler.
    """

    text: str
    pcm: bytes
    sample_rate: int
    sample_width: int  # bytes per sample
    channels: int

    @property
    def frame_size(self) -> int:
        """Bytes per frame across all channels."""
        return self.sample_width * self.channels

    @property
    def duration(self) -> float:
        """Measured duration in seconds."""
        if not self.frame_size or not self.sample_rate:
            return 0.0
        return (len(self.pcm) / self.frame_size) / self.sample_rate

    def matches_format(self, other: "SynthesizedSegment") -> bool:
        """Whether two segments can be concatenated without resampling."""
        return (
            self.sample_rate == other.sample_rate
            and self.sample_width == other.sample_width
            and self.channels == other.channels
        )


@dataclass(frozen=True)
class TextSegment:
    """Represents a segment of text with timing information."""

    text: str
    start_time: float  # seconds from beginning
    duration: float  # segment duration in seconds
    segment_type: str  # "sentence", "paragraph", "heading"
    chunk_index: int  # which audio chunk this belongs to
    sentence_index: int  # position within the chunk

    @property
    def end_time(self) -> float:
        """Return end time of the text segment."""
        return self.start_time + self.duration


@dataclass(frozen=True)
class TimingMetadata:
    """Complete timing information for a document."""

    total_duration: float
    text_segments: list[TextSegment]
    audio_files: list[str]  # just filenames for now


@dataclass(frozen=True)
class TimedAudioResult:
    """Audio generation result with optional timing data."""

    audio_files: list[str]
    combined_mp3: str | None
    timing_data: TimingMetadata | None = None

    @property
    def has_timing_data(self) -> bool:
        """Return True if timing data is available."""
        return self.timing_data is not None
