"""Thread-safe progress tracking for long-running operations.

This module provides a thread-safe implementation for tracking progress
of document processing operations. Designed for single-user home app
but with proper locking to prevent race conditions.

The store is a regular service: it is created by Services (application/composition.py)
and reaches consumers via constructor injection (DocumentProcessingService) or
the Services instance on the Flask app (routes). There is no module-level instance.
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressStatus:
    """Progress tracking for long-running operations."""

    operation_id: str
    stage: str
    percentage: int
    message: str
    is_complete: bool = False
    is_error: bool = False
    error_message: str | None = None
    result_data: dict[str, Any] | None = None
    cancelled: bool = False


class ThreadSafeProgressStore:
    """Thread-safe storage for operation progress tracking.

    Uses a lock to ensure thread-safety for concurrent access.
    Includes automatic cleanup of stale entries.
    """

    def __init__(self, max_age_seconds: int = 3600, max_entries: int = 100):
        """Initialize the progress store.

        Args:
            max_age_seconds: Maximum age of entries before cleanup (default 1 hour)
            max_entries: Maximum number of entries before forced cleanup
        """
        self._lock = threading.Lock()
        self._progress: dict[str, ProgressStatus] = {}
        self._cancellation_flags: dict[str, bool] = {}
        self._timestamps: dict[str, float] = {}
        self._max_age_seconds = max_age_seconds
        self._max_entries = max_entries

    def update(
        self,
        operation_id: str,
        stage: str,
        percentage: int,
        message: str,
        is_complete: bool = False,
        is_error: bool = False,
        error_message: str | None = None,
        result_data: dict[str, Any] | None = None,
    ) -> None:
        """Update progress for an operation (thread-safe).

        If the operation has been cancelled, the update is ignored to prevent
        a racing background thread from overwriting the cancellation status.
        """
        logger.debug("Progress [%s]: stage=%s, pct=%d%%, msg=%s", operation_id[:8], stage, percentage, message)
        with self._lock:
            # Don't overwrite cancellation status with a regular update
            if self._cancellation_flags.get(operation_id, False) and not is_complete and not is_error:
                logger.debug("Progress [%s]: update ignored, operation is cancelled", operation_id[:8])
                return

            self._progress[operation_id] = ProgressStatus(
                operation_id=operation_id,
                stage=stage,
                percentage=percentage,
                message=message,
                is_complete=is_complete,
                is_error=is_error,
                error_message=error_message,
                result_data=result_data,
                cancelled=self._cancellation_flags.get(operation_id, False),
            )
            self._timestamps[operation_id] = time.time()
            self._cleanup_if_needed()

    def get(self, operation_id: str) -> ProgressStatus | None:
        """Get current progress for an operation (thread-safe)."""
        with self._lock:
            return self._progress.get(operation_id)

    def cancel(self, operation_id: str) -> None:
        """Mark an operation as cancelled (thread-safe)."""
        logger.info("Operation cancelled: %s", operation_id[:8])
        with self._lock:
            self._cancellation_flags[operation_id] = True
            self._timestamps[operation_id] = time.time()
            if operation_id in self._progress:
                current = self._progress[operation_id]
                self._progress[operation_id] = ProgressStatus(
                    operation_id=current.operation_id,
                    stage="cancelled",
                    percentage=current.percentage,
                    message="Processing cancelled by user",
                    is_complete=True,
                    is_error=False,
                    cancelled=True,
                )

    def is_cancelled(self, operation_id: str) -> bool:
        """Check if an operation has been cancelled (thread-safe)."""
        with self._lock:
            return self._cancellation_flags.get(operation_id, False)

    def remove(self, operation_id: str) -> None:
        """Remove an operation from tracking (thread-safe)."""
        with self._lock:
            self._remove_locked(operation_id)

    def _remove_locked(self, operation_id: str) -> None:
        """Remove all state for an operation (must be called with lock held)."""
        self._progress.pop(operation_id, None)
        self._cancellation_flags.pop(operation_id, None)
        self._timestamps.pop(operation_id, None)

    def _cleanup_if_needed(self) -> None:
        """Evict stale finished entries, then oldest entries beyond the cap (must be called with lock held).

        In-flight operations are exempt from age-based eviction: a long processing
        phase emits no updates, and evicting the entry would drop its cancellation
        flag while the background thread is still running.
        """
        current_time = time.time()
        stale_ids = [
            op_id
            for op_id, timestamp in self._timestamps.items()
            if current_time - timestamp > self._max_age_seconds and self._is_finished(op_id)
        ]
        for op_id in stale_ids:
            self._remove_locked(op_id)
            logger.debug("Cleaned up stale progress entry: %s", op_id)

        overflow = len(self._progress) - self._max_entries
        if overflow > 0:
            # Keep the cap hard, but sacrifice finished entries before in-flight ones
            eviction_order = sorted(
                self._timestamps, key=lambda op_id: (not self._is_finished(op_id), self._timestamps[op_id])
            )
            for op_id in eviction_order[:overflow]:
                self._remove_locked(op_id)
                logger.debug("Evicted oldest progress entry over cap: %s", op_id)

    def _is_finished(self, operation_id: str) -> bool:
        """Whether an operation is complete or errored (must be called with lock held)."""
        status = self._progress.get(operation_id)
        return status is None or status.is_complete or status.is_error
