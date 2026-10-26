# application/config/file_configs.py
"""File handling and cleanup configuration classes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FileConfig:
    """File handling configuration."""

    upload_folder: str = "uploads"
    audio_folder: str = "audio_outputs"
    max_file_size_mb: int = 100


@dataclass(frozen=True)
class FileCleanupConfig:
    """File cleanup configuration."""

    enabled: bool = True
    max_file_age_hours: float = 24.0
    auto_cleanup_interval_hours: float = 6.0
