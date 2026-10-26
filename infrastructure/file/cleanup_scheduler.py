"""Background service that periodically removes old files.

Runs in a separate thread and delegates the actual sweep to
IFileManager.cleanup_old_files, which deletes by modification time.
"""

import logging
import threading

from domain.interfaces import IFileManager

logger = logging.getLogger(__name__)


class FileCleanupScheduler:
    """Background thread that sweeps files older than a maximum age on a fixed interval."""

    def __init__(self, file_manager: IFileManager, max_file_age_hours: float, interval_seconds: float):
        """Initialize the cleanup scheduler (does not start automatically).

        Args:
            file_manager: IFileManager implementation for file operations
            max_file_age_hours: Files older than this are deleted
            interval_seconds: How often to run a sweep
        """
        self.file_manager = file_manager
        self.max_file_age_hours = max_file_age_hours
        self.interval_seconds = interval_seconds

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background cleanup thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._cleanup_job, daemon=True, name="verbatimpapers-cleanup")
        self._thread.start()
        logger.info(
            "File cleanup scheduler started (interval=%.0fs, max_age=%.1fh)",
            self.interval_seconds,
            self.max_file_age_hours,
        )

    def stop(self) -> None:
        """Stop the background cleanup thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.info("Stopping file cleanup scheduler")
            self._stop_event.set()
            self._thread.join(timeout=5)

    def run_once(self) -> None:
        """Run a single sweep, logging the outcome. Never raises."""
        try:
            result = self.file_manager.cleanup_old_files(self.max_file_age_hours)
        except Exception:
            logger.exception("Error during cleanup cycle")
            return

        if result.is_failure or result.value is None:
            logger.error("Cleanup cycle failed: %s", result.error)
            return

        stats = result.value
        for error in stats.errors:
            logger.warning("Cleanup: %s", error)
        if stats.files_removed:
            logger.info("Cleanup removed %d files (%d bytes)", stats.files_removed, stats.bytes_freed)
        else:
            logger.debug("Cleanup cycle: nothing to remove")

    def _cleanup_job(self) -> None:
        """Main loop: sweep immediately, then once per interval until stopped."""
        while not self._stop_event.is_set():
            self.run_once()
            self._stop_event.wait(self.interval_seconds)
