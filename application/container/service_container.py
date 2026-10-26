# application/container/service_container.py - Concrete Service Container Implementation
"""Concrete service container that wires together all dependencies.

Lives in the application layer because it imports from infrastructure/ and application/,
which would violate hexagonal architecture if placed in domain/.
The domain layer only contains the IServiceContainer interface.
"""

from collections.abc import Callable
import logging
import threading
import types
from typing import TypeVar

from application.config.system_config import SystemConfig
from domain.container.service_container import IServiceContainer, ServiceFactory, ServiceKey

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceContainer(IServiceContainer):
    """Immutable service container with lazy initialization.

    High cohesion: All dependency injection in one place.
    Low coupling: Uses factories to avoid tight dependencies.
    Thread-safe: Uses immutable MappingProxyType for factories.
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config

        # Build all factories upfront (immutable)
        factories = self._build_core_services()
        self._factories: types.MappingProxyType[ServiceKey, ServiceFactory] = types.MappingProxyType(factories)
        logger.debug("ServiceContainer initialized with %d factories", len(factories))

        # Mutable singleton cache (internal implementation detail)
        self._singletons: dict[ServiceKey, object] = {}
        self._lock = threading.RLock()

    def get(self, interface: type[T] | str) -> T:
        """Get service instance (singleton pattern, thread-safe)."""
        interface_name = getattr(interface, "__name__", str(interface))
        if interface in self._singletons:
            logger.debug("Service resolved (cached): %s", interface_name)
            return self._singletons[interface]  # type: ignore[return-value]

        with self._lock:
            # Double-check after acquiring lock
            if interface in self._singletons:
                logger.debug("Service resolved (cached after lock): %s", interface_name)
                return self._singletons[interface]  # type: ignore[return-value]

            if interface not in self._factories:
                raise ValueError(f"Service {interface_name} not registered")

            logger.debug("Creating service: %s", interface_name)
            instance = self._factories[interface]()
            self._singletons[interface] = instance
            logger.debug("Service created: %s -> %s", interface_name, type(instance).__name__)
            return instance  # type: ignore[return-value]

    def has(self, interface: type[T] | str) -> bool:
        """Check if service is registered."""
        return interface in self._factories

    def _build_core_services(self) -> dict[ServiceKey, ServiceFactory]:
        """Build all core service factories upfront (immutable pattern)."""
        from application.services.document_service import DocumentProcessingService
        from application.services.progress_store import ThreadSafeProgressStore
        from domain.audio.audio_assembler import AudioAssembler
        from domain.audio.audio_engine import AudioEngine
        from domain.audio.duration_measurer import AudioDurationMeasurer
        from domain.audio.timing_engine import ITimingEngine, TimingEngine
        from domain.document.document_engine import DocumentEngine
        from domain.interfaces import (
            IAudioDurationMeasurer,
            IAudioEngine,
            IDocumentEngine,
            IFileManager,
            ILLMProvider,
            IPdfTextExtractor,
            ITextPipeline,
        )
        from domain.text.text_pipeline import TextPipeline
        from infrastructure.file.file_manager import FileManager
        from infrastructure.llm.gemini_llm_provider import GeminiLLMProvider
        from infrastructure.ocr.tesseract_ocr_provider import TesseractOCRProvider
        from infrastructure.pdf.pdfplumber_text_extractor import PdfPlumberTextExtractor

        # Build all factories in a single immutable dict
        factories: dict[ServiceKey, ServiceFactory] = {
            # Application Services
            ThreadSafeProgressStore: lambda: ThreadSafeProgressStore(),
            DocumentProcessingService: lambda: DocumentProcessingService(
                service_container=self, config=self.config, progress_store=self.get(ThreadSafeProgressStore)
            ),
            # File Manager
            IFileManager: lambda: FileManager(
                upload_folder=self.config.files.upload_folder, output_folder=self.config.files.audio_folder
            ),
            FileManager: lambda: self.get(IFileManager),
            # TTS Engine (factory method)
            "tts_engine": lambda: self._create_tts_engine(),
            # Audio Duration Measurer (breaks circular dependency between TimingEngine and AudioEngine)
            IAudioDurationMeasurer: lambda: AudioDurationMeasurer(),
            # Text Pipeline
            ITextPipeline: lambda: TextPipeline(
                llm_provider=self.get(GeminiLLMProvider) if self.config.gemini and self.config.gemini.api_key else None,
                enable_cleaning=self.config.text_processing.enable_cleaning,
            ),
            # Plain English variant of TextPipeline
            "plain_english_text_pipeline": lambda: TextPipeline(
                llm_provider=self.get(GeminiLLMProvider) if self.config.gemini and self.config.gemini.api_key else None,
                enable_cleaning=self.config.text_processing.enable_cleaning,
                enable_plain_english=True,
            ),
            # Audio Assembler - single owner of gap policy and container writing.
            # The gap is engine-independent policy; it reads from the Piper block only
            # because that is where the setting has always lived.
            AudioAssembler: lambda: AudioAssembler(
                sentence_gap=self.config.piper.sentence_silence if self.config.piper else 0.2
            ),
            # Timing Engine - shares the assembler so the timeline matches the audio
            ITimingEngine: lambda: TimingEngine(
                tts_engine=self.get("tts_engine"),
                file_manager=self.get(FileManager),
                assembler=self.get(AudioAssembler),
                request_interval=(
                    self.config.tts.request_delay_seconds if self.config.tts.engine.value == "gemini" else 0.0
                ),
            ),
            # Audio Engine
            IAudioEngine: lambda: AudioEngine(
                tts_engine=self.get("tts_engine"),
                file_manager=self.get(FileManager),
                timing_engine=self.get(ITimingEngine),
                duration_measurer=self.get(IAudioDurationMeasurer),
                assembler=self.get(AudioAssembler),
                max_concurrent=self.config.performance.audio_concurrent_chunks,
                audio_target_chunk_size=self.config.text_processing.audio_target_chunk_size,
                enable_async=self.config.performance.enable_async_audio,
            ),
            # Document Engine
            IDocumentEngine: lambda: DocumentEngine(
                ocr_provider=self.get(TesseractOCRProvider),
                file_manager=self.get(FileManager),
                pdf_extractor=self.get(IPdfTextExtractor),
                audio_target_chunk_size=self.config.text_processing.audio_target_chunk_size,
                ocr_resolution=self.config.ocr.dpi,
            ),
            # PDF Text Extractor
            IPdfTextExtractor: lambda: PdfPlumberTextExtractor(),
            # OCR Provider
            TesseractOCRProvider: lambda: TesseractOCRProvider(language=self.config.ocr.language),
        }

        logger.debug(
            "Registered %d core service factories (LLM configured: %s)",
            len(factories),
            bool(self.config.gemini and self.config.gemini.api_key),
        )

        # Add LLM Provider only if configured
        if self.config.gemini and self.config.gemini.api_key:
            api_key = self.config.gemini.api_key
            if api_key is not None:  # Type guard for mypy
                # Text cleaning uses llm.model_name only; tts.gemini.model_name is a
                # speech-only model and must never be used for text tasks
                llm = self.config.llm
                factories[GeminiLLMProvider] = lambda: GeminiLLMProvider(
                    model_name=llm.model_name,
                    api_key=api_key,
                    min_request_interval=llm.request_delay_seconds,
                    max_concurrent_requests=llm.concurrent_requests,
                )
                factories[ILLMProvider] = lambda: self.get(GeminiLLMProvider)

        return factories

    def _create_tts_engine(self) -> object:  # Returns TTS engine instance
        """Factory for TTS engine based on configuration."""
        from infrastructure.tts.gemini_tts_provider import GeminiTTSProvider
        from infrastructure.tts.piper_tts_provider import PiperTTSProvider

        logger.info("Creating TTS engine: %s", self.config.tts.engine.value)
        if self.config.tts.engine.value == "gemini":
            if not self.config.gemini or not self.config.gemini.api_key:
                raise ValueError("Gemini API key is required for Gemini TTS engine")
            return GeminiTTSProvider(
                model_name=self.config.gemini.model_name,
                api_key=self.config.gemini.api_key,
                voice_name=self.config.gemini.voice_name,
                style_prompt=self.config.gemini.style_prompt,
                min_request_interval=self.config.tts.request_delay_seconds,
                max_concurrent_requests=self.config.tts.concurrent_requests,
                requests_per_minute=30,  # Default rate limit for Flash TTS
            )
        else:
            piper_config = self.config.piper
            if piper_config is None:
                from domain.config.tts_config import PiperConfig

                piper_config = PiperConfig()

            return PiperTTSProvider(
                config=piper_config,
                repository_url=piper_config.model_repository_url,
                project_root=self.config.project_root,
            )


class ImmutableServiceContainerBuilder:
    """Builder for creating immutable service containers with additional registrations."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self._additional_factories: dict[ServiceKey, ServiceFactory] = {}

    def register(self, interface: type[T] | str, factory: Callable[[], T]) -> "ImmutableServiceContainerBuilder":
        """Register additional service factory (builder pattern)."""
        self._additional_factories[interface] = factory
        return self

    def build(self) -> ServiceContainer:
        """Build the immutable service container."""
        # Create container with core services
        container = ServiceContainer(self.config)

        # Add additional factories if any (they'll be merged into the immutable proxy)
        if self._additional_factories:
            # Get existing factories and merge with additional ones
            all_factories = dict(container._factories)  # Convert proxy to dict
            all_factories.update(self._additional_factories)

            # Replace with new immutable proxy
            object.__setattr__(container, "_factories", types.MappingProxyType(all_factories))

        return container


def create_service_container(config: SystemConfig) -> ServiceContainer:
    """Factory function to create configured service container."""
    return ImmutableServiceContainerBuilder(config).build()


def create_service_container_builder(config: SystemConfig) -> ImmutableServiceContainerBuilder:
    """Factory function to create service container builder for additional registrations."""
    return ImmutableServiceContainerBuilder(config)
