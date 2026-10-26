"""Composition root: builds the application's object graph from SystemConfig.

Every dependency is a typed, cached property, so mypy checks the wiring and each
service is created once, on first access. create_app calls warm_up() before the
app serves requests, so request threads only ever read already-built services.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
import logging

from application.config.system_config import SystemConfig
from application.services.document_service import DocumentProcessingService
from application.services.progress_store import ThreadSafeProgressStore
from domain.audio.audio_assembler import AudioAssembler
from domain.audio.audio_engine import AudioEngine
from domain.config.tts_config import PiperConfig, TTSEngine
from domain.document.document_engine import DocumentEngine
from domain.errors import Result, configuration_error
from domain.interfaces import (
    IAudioEncoder,
    IAudioEngine,
    IDocumentEngine,
    IFileManager,
    ILLMProvider,
    IOCRProvider,
    IPdfTextExtractor,
    ITextPipeline,
    ITTSEngine,
)
from domain.text.text_pipeline import TextPipeline
from infrastructure.audio.ffmpeg_audio_encoder import FfmpegAudioEncoder
from infrastructure.file.cleanup_scheduler import FileCleanupScheduler
from infrastructure.file.file_manager import FileManager
from infrastructure.llm.gemini_llm_provider import GeminiLLMProvider
from infrastructure.ocr.tesseract_ocr_provider import TesseractOCRProvider
from infrastructure.pdf.pdfplumber_text_extractor import PdfPlumberTextExtractor
from infrastructure.tts.gemini_tts_provider import GeminiTTSProvider
from infrastructure.tts.piper_tts_provider import PiperTTSProvider

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """The application's services, wired from one SystemConfig."""

    config: SystemConfig

    def warm_up(self) -> Result[None]:
        """Build every service now and check the TTS engine can synthesize.

        Run once at startup: the Piper model download happens here rather than on
        the first upload, and a broken engine is reported before any request.
        """
        try:
            _ = self.document_service, self.background_executor, self.cleanup_scheduler
        except Exception as e:
            logger.exception("Service construction failed")
            return Result.failure(configuration_error(f"Could not build services: {e}"))
        return self.tts_engine.check_ready()

    # === Application services ===

    @cached_property
    def progress_store(self) -> ThreadSafeProgressStore:
        """Progress of background operations, shared by routes and workers."""
        return ThreadSafeProgressStore()

    @cached_property
    def document_service(self) -> DocumentProcessingService:
        """Orchestrates one upload's conversion in the background."""
        return DocumentProcessingService(
            document_engine=self.document_engine,
            audio_engine=self.audio_engine,
            text_pipeline=self.text_pipeline,
            plain_english_text_pipeline=self.plain_english_text_pipeline,
            progress_store=self.progress_store,
            config=self.config,
        )

    @cached_property
    def background_executor(self) -> ThreadPoolExecutor:
        """Bounded worker pool for document processing; uploads beyond the cap queue up."""
        return ThreadPoolExecutor(
            max_workers=self.config.performance.max_concurrent_operations,
            thread_name_prefix="verbatimpapers-worker",
        )

    @cached_property
    def cleanup_scheduler(self) -> FileCleanupScheduler | None:
        """Periodic output cleanup, or None when disabled. Created stopped."""
        if not self.config.cleanup.enabled:
            return None
        return FileCleanupScheduler(
            file_manager=self.file_manager,
            max_file_age_hours=self.config.cleanup.max_file_age_hours,
            interval_seconds=self.config.cleanup.auto_cleanup_interval_hours * 3600,
        )

    # === Domain engines ===

    @cached_property
    def document_engine(self) -> IDocumentEngine:
        """PDF text extraction, with OCR fallback."""
        return DocumentEngine(
            ocr_provider=self.ocr_provider,
            pdf_extractor=self.pdf_extractor,
            ocr_resolution=self.config.ocr.dpi,
        )

    @cached_property
    def audio_engine(self) -> IAudioEngine:
        """Turns text chunks into one MP3 plus its read-along timeline."""
        return AudioEngine(
            tts_engine=self.tts_engine,
            file_manager=self.file_manager,
            encoder=self.audio_encoder,
            assembler=self.audio_assembler,
            max_concurrent=self.config.performance.audio_concurrent_chunks,
            audio_target_chunk_size=self.config.text_processing.audio_target_chunk_size,
        )

    @cached_property
    def audio_assembler(self) -> AudioAssembler:
        """Single owner of the inter-sentence gap and container writing."""
        return AudioAssembler(sentence_gap=self.config.audio.sentence_gap)

    @cached_property
    def text_pipeline(self) -> ITextPipeline:
        """Text cleaning for the default conversion."""
        return TextPipeline(llm_provider=self.llm_provider, enable_cleaning=self.config.text_processing.enable_cleaning)

    @cached_property
    def plain_english_text_pipeline(self) -> ITextPipeline:
        """Text cleaning plus plain-English rewriting."""
        return TextPipeline(
            llm_provider=self.llm_provider,
            enable_cleaning=self.config.text_processing.enable_cleaning,
            enable_plain_english=True,
        )

    # === Infrastructure adapters ===

    @cached_property
    def file_manager(self) -> IFileManager:
        """Upload and output file storage."""
        return FileManager(upload_folder=self.config.files.upload_folder, output_folder=self.config.files.audio_folder)

    @cached_property
    def audio_encoder(self) -> IAudioEncoder:
        """MP3 encoding via ffmpeg."""
        return FfmpegAudioEncoder()

    @cached_property
    def ocr_provider(self) -> IOCRProvider:
        """Tesseract OCR for scanned pages."""
        return TesseractOCRProvider(language=self.config.ocr.language)

    @cached_property
    def pdf_extractor(self) -> IPdfTextExtractor:
        """Text-layer extraction from PDFs."""
        return PdfPlumberTextExtractor()

    @cached_property
    def llm_provider(self) -> ILLMProvider | None:
        """Text-cleaning LLM, or None without an API key (cleaning is then skipped)."""
        gemini = self.config.gemini
        if gemini is None or not gemini.api_key:
            return None
        # Text cleaning uses llm.model_name only; tts.gemini.model_name is a
        # speech-only model and must never be used for text tasks
        return GeminiLLMProvider(model_name=self.config.llm.model_name, api_key=gemini.api_key)

    @cached_property
    def tts_engine(self) -> ITTSEngine:
        """The configured speech engine: Piper (local) or Gemini (cloud)."""
        logger.info("Creating TTS engine: %s", self.config.tts.engine.value)
        if self.config.tts.engine == TTSEngine.GEMINI:
            gemini = self.config.gemini
            if gemini is None or not gemini.api_key:
                raise ValueError("Gemini TTS needs secrets.google_ai_api_key in config.yaml")
            return GeminiTTSProvider(
                model_name=gemini.model_name,
                api_key=gemini.api_key,
                voice_name=gemini.voice_name,
                style_prompt=gemini.style_prompt,
                min_request_interval=self.config.tts.request_delay_seconds,
                max_concurrent_requests=self.config.tts.concurrent_requests,
            )
        piper = self.config.piper or PiperConfig()
        return PiperTTSProvider(
            config=piper, repository_url=piper.model_repository_url, project_root=self.config.project_root
        )
