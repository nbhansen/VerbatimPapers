# application/config/logging_config.py
from dataclasses import dataclass
import datetime
import json
import logging
import logging.handlers
from pathlib import Path
import sys


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for application logging system."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str | None = None
    json_format: bool = False
    max_file_size_mb: int = 10
    backup_count: int = 5
    console_output: bool = True

    def __post_init__(self) -> None:
        """Validate logging configuration."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")

        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")

        if self.backup_count < 0:
            raise ValueError("backup_count cannot be negative")


class JSONFormatter(logging.Formatter):
    """One JSON object per record, for structured log collection."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the record as a single-line JSON object."""
        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


# Handlers installed by setup_logging carry this name prefix, so reconfiguring
# replaces only ours and leaves foreign handlers (e.g. pytest's capture) alone.
_HANDLER_PREFIX = "verbatimpapers."

# Third-party libraries that flood the log at DEBUG/INFO.
_NOISY_LOGGERS = ("pdfminer", "pdfplumber", "PIL", "urllib3", "httpx", "httpcore", "google_genai")


def setup_logging(config: LoggingConfig) -> None:
    """Configure the root logger, so every `logging.getLogger(__name__)` logger is covered.

    Safe to call more than once: handlers from a previous call are replaced.
    """
    root = logging.getLogger()

    for handler in [h for h in root.handlers if (h.get_name() or "").startswith(_HANDLER_PREFIX)]:
        root.removeHandler(handler)
        handler.close()

    root.setLevel(config.level.upper())
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    formatter = JSONFormatter() if config.json_format else logging.Formatter(config.format)

    handlers: list[logging.Handler] = []
    if config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.set_name(_HANDLER_PREFIX + "console")
        handlers.append(console_handler)

    if config.file_path:
        Path(config.file_path).parent.mkdir(parents=True, exist_ok=True)
        # Rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.set_name(_HANDLER_PREFIX + "file")
        handlers.append(file_handler)

    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
