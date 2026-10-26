# application/config/file_configs.py
"""File handling and cleanup configuration classes."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FileConfig:
    """File handling configuration."""

    upload_folder: str = "uploads"
    audio_folder: str = "audio_outputs"
    max_file_size_mb: int = 100
    allowed_extensions: frozenset[str] = field(default_factory=lambda: frozenset({"pdf"}))
    audio_extensions: frozenset[str] = field(default_factory=lambda: frozenset({"wav", "mp3"}))


@dataclass(frozen=True)
class FileCleanupConfig:
    """File cleanup configuration."""

    enabled: bool = True
    max_file_age_hours: float = 24.0
    auto_cleanup_interval_hours: float = 6.0
