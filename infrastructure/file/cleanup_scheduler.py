"""Background service for managing cleanup of old files.

Runs in a separate thread to periodically remove expired files.
"""

import logging
import threading
import time

from domain.interfaces import IFileManager

logger = logging.getLogger(__name__)


class FileCleanupScheduler:
    """Background thread service for periodic cleanup of expired files.

    Monitors registered files and removes them when they exceed max age.
    """

    def __init__(self, file_manager: IFileManager, max_file_age_seconds: int, check_interval_seconds: int):
        """Initialize the cleanup scheduler (does not start automatically).

        Args:
            file_manager: IFileManager implementation for file operations
            max_file_age_seconds: Maximum file age before deletion
            check_interval_seconds: How often to check for expired files
        """
        self.file_manager = file_manager
        self.max_file_age_seconds = max_file_age_seconds
        self.check_interval_seconds = check_interval_seconds

        # Thread-safe file tracking
        self._lock = threading.Lock()
        self._scheduled_files: dict[str, float] = {}  # {filepath: creation_timestamp}

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def schedule(self, filepath: str) -> None:
        """Schedule a file for cleanup monitoring."""
        logger.debug("Scheduling file for cleanup: %s", filepath)
        with self._lock:
            self._scheduled_files[filepath] = time.time()

    def start(self) -> None:
        """Start the background cleanup thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._cleanup_job, daemon=True)
        self._thread.start()
        logger.info(
            "File cleanup scheduler started (interval=%ds, max_age=%ds)",
            self.check_interval_seconds,
            self.max_file_age_seconds,
        )

    def stop(self) -> None:
        """Stop the background cleanup thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.info("Stopping file cleanup scheduler")
            self._stop_event.set()
            self._thread.join(timeout=5)

    def run_manual_cleanup(self) -> dict[str, int]:
        """Run one cleanup pass immediately and report how many tracked files were deleted."""
        logger.info("Manual cleanup triggered")
        return {"files_deleted": self._process_expired_files()}

    def _cleanup_job(self) -> None:
        """Main cleanup loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                logger.debug("Cleanup cycle: checking for expired files")
                self._process_expired_files()
            except Exception:
                logger.exception("Error during cleanup cycle")

            # Wait for next interval, checking stop event
            self._stop_event.wait(self.check_interval_seconds)

    def _process_expired_files(self) -> int:
        """Process and remove expired files, returning the number deleted."""
        current_time = time.time()

        # Collect expired files under lock (fast), then delete outside lock (slow I/O)
        with self._lock:
            logger.debug("Checking %d tracked files for expiration", len(self._scheduled_files))
            expired = {
                filepath: creation_time
                for filepath, creation_time in self._scheduled_files.items()
                if (current_time - creation_time) > self.max_file_age_seconds
            }

        if not expired:
            return 0

        # Delete files outside lock so schedule() isn't blocked by I/O
        successfully_deleted = []
        for filepath, creation_time in expired.items():
            try:
                self.file_manager.delete_file(filepath)
                logger.info("Deleted expired file: %s (age=%.0fs)", filepath, current_time - creation_time)
                successfully_deleted.append(filepath)
            except Exception:
                logger.exception("Failed to delete expired file: %s (will retry next cycle)", filepath)

        # Remove only successfully deleted files from tracking
        if successfully_deleted:
            with self._lock:
                for filepath in successfully_deleted:
                    self._scheduled_files.pop(filepath, None)
            logger.debug("Cleanup cycle complete: %d files deleted", len(successfully_deleted))
        return len(successfully_deleted)
