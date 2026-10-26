"""Concrete implementation of IFileManager for local file system operations.

Handles file I/O, directory management, and temporary file creation.
"""

import logging
import math
import os
from pathlib import Path
import tempfile
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

    def get_upload_dir(self) -> str:
        """Returns the absolute path to the upload directory."""
        return self.upload_folder

    def save_temp_file(self, content: bytes, suffix: str = ".tmp") -> str:
        """Saves content to a temporary file and returns its full path."""
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.output_folder)
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(content)
        logger.debug("Saved temp file: %s (%d bytes, suffix=%s)", path, len(content), suffix)
        return path

    def save_output_file(self, content: bytes, filename: str) -> str:
        """Saves content to a final output file and returns its full path."""
        if not filename:
            raise ValueError("Filename cannot be empty")

        # Sanitize filename to prevent directory traversal
        base_filename = Path(filename).name
        output_path = Path(self.output_folder) / base_filename

        output_path.write_bytes(content)
        logger.debug("Saved output file: %s (%d bytes)", base_filename, len(content))

        return str(output_path)

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

    def delete_file(self, filepath: str) -> None:
        """Deletes a file if it exists."""
        # Security check: ensure path is within managed directories
        abs_path = Path(filepath).resolve()
        if not (abs_path.is_relative_to(self.output_folder) or abs_path.is_relative_to(self.upload_folder)):
            logger.error("Attempted file deletion outside managed directories: %s", filepath)
            raise ValueError(f"Cannot delete file outside managed directories: {filepath}")

        if abs_path.exists():
            abs_path.unlink()
            logger.debug("Deleted file: %s", filepath)
        else:
            logger.warning("File not found for deletion: %s", filepath)
