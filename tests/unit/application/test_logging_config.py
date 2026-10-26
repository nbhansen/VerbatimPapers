"""Logging setup must cover every module logger, not just one app namespace."""

from collections.abc import Iterator
import logging
from pathlib import Path

import pytest

from application.config.logging_config import LoggingConfig, setup_logging


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    """Undo setup_logging's changes to the root logger after each test."""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    for handler in root.handlers:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


@pytest.mark.usefixtures("restore_root_logger")
class TestSetupLogging:
    """setup_logging configures the root logger for all module loggers."""

    def test_module_logger_reaches_file_at_configured_level(self, tmp_path: Path) -> None:
        """A plain __name__ logger in domain/ or infrastructure/ is written out."""
        log_file = tmp_path / "app.log"
        setup_logging(LoggingConfig(level="DEBUG", file_path=str(log_file), console_output=False))

        logging.getLogger("domain.audio.audio_engine").debug("domain debug line")
        logging.getLogger("infrastructure.tts.piper_tts_provider").info("adapter info line")

        content = log_file.read_text(encoding="utf-8")
        assert "domain.audio.audio_engine - DEBUG - domain debug line" in content
        assert "adapter info line" in content

    def test_records_below_configured_level_are_dropped(self, tmp_path: Path) -> None:
        """The configured level applies to module loggers too."""
        log_file = tmp_path / "app.log"
        setup_logging(LoggingConfig(level="WARNING", file_path=str(log_file), console_output=False))

        logging.getLogger("application.services.document_service").info("too quiet")

        assert "too quiet" not in log_file.read_text(encoding="utf-8")

    def test_reconfiguring_replaces_own_handlers_and_keeps_foreign_ones(self, tmp_path: Path) -> None:
        """Repeated setup does not stack handlers or remove ones it did not install."""
        root = logging.getLogger()
        foreign = logging.NullHandler()
        root.addHandler(foreign)

        setup_logging(LoggingConfig(file_path=str(tmp_path / "a.log")))
        setup_logging(LoggingConfig(file_path=str(tmp_path / "b.log")))

        own = [h for h in root.handlers if (h.get_name() or "").startswith("verbatimpapers.")]
        assert len(own) == 2  # one console + one file, not duplicated
        assert foreign in root.handlers
