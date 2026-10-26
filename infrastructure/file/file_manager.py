"""Concrete implementation of IFileManager for local file system operations.

Handles file I/O, directory management, and temporary file creation.
"""

import logging
import math
from pathlib import Path
import time

from domain.errors import Result, configuration_error
from domain.interfaces import IFileManager
from domain.models import CleanupStats

logger = logging.getLogger(__name__)


class FileManager(IFileManager):
    """Manages file I/O, directory paths, and temporary file creation."""

    def __init__(self, upload_folder: str, output_folder: str):
        self.upload_folder = str(Path(upload_folder).resolve())
        self.output_folder = str(Path(output_folder).resolve())
        Path(self.upload_folder).mkdir(parents=True, exist_ok=True)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        logger.debug("FileManager initialized: upload=%s, output=%s", self.upload_folder, self.output_folder)

    def get_output_dir(self) -> str:
        """Returns the absolute path to the output directory."""
        return self.output_folder

    def cleanup_old_files(self, max_age_hours: float) -> Result[CleanupStats]:
        """Deletes files in the output and upload directories with mtime older than max_age_hours.

        Uploads are normally removed when processing finishes; sweeping them too
        catches the ones a crash mid-processing leaves behind.

        Rejects non-positive and non-finite ages: a negative age would put the
        cutoff in the future and delete every file, and NaN comparisons would
        silently match nothing.
        """
        if not math.isfinite(max_age_hours) or max_age_hours <= 0:
            return Result.failure(
                configuration_error(f"max_age_hours must be a positive finite number, got {max_age_hours!r}")
            )

        cutoff_time = time.time() - max_age_hours * 3600
        files_removed = 0
        bytes_freed = 0
        errors: list[str] = []

        candidates = [path for folder in (self.output_folder, self.upload_folder) for path in Path(folder).iterdir()]
        for filepath in candidates:
            try:
                stat = filepath.stat()
                if not filepath.is_file() or stat.st_mtime >= cutoff_time:
                    continue
                filepath.unlink()
                files_removed += 1
                bytes_freed += stat.st_size
                logger.info("Cleaned up old file: %s", filepath.name)
            except OSError as e:
                errors.append(f"Failed to remove {filepath.name}: {e!s}")

        return Result.success(CleanupStats(files_removed=files_removed, bytes_freed=bytes_freed, errors=tuple(errors)))
