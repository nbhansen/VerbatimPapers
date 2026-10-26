"""Application context for dependency injection and global state elimination.

This module provides the ApplicationContext class which encapsulates all application-wide
dependencies and eliminates global state, following hexagonal architecture principles.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging

from application.config.system_config import SystemConfig
from application.container.service_container import ServiceContainer
from application.services.progress_store import ThreadSafeProgressStore
from infrastructure.file.cleanup_scheduler import FileCleanupScheduler


@dataclass(frozen=True)
class ApplicationContext:
    """Immutable application context that encapsulates all application-wide dependencies.

    This eliminates global state by providing a single, thread-safe container
    for all application services and configuration.
    """

    config: SystemConfig
    service_container: ServiceContainer
    progress_store: ThreadSafeProgressStore
    background_executor: ThreadPoolExecutor
    cleanup_scheduler: FileCleanupScheduler | None = None

    @property
    def is_processor_available(self) -> bool:
        """Check if PDF processing service is available."""
        logger = logging.getLogger(__name__)
        try:
            # Check if we can get an audio engine (main processing component)
            from domain.interfaces import IAudioEngine

            audio_service = self.service_container.get(IAudioEngine)
            result = audio_service is not None
            logger.debug("Audio service check - service=%s, result=%s", audio_service, result)
            return result
        except Exception as e:
            logger.debug("Audio service check failed: %s", e)
            return False
