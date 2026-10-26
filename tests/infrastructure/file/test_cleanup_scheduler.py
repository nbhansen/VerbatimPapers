# tests/infrastructure/file/test_cleanup_scheduler.py
"""Tests for FileCleanupScheduler: a periodic loop around IFileManager.cleanup_old_files."""

import os
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock

import pytest

from app_factory import is_serving_process
from domain.errors import Result, configuration_error
from domain.interfaces import IFileManager
from domain.models import CleanupStats
from infrastructure.file.cleanup_scheduler import FileCleanupScheduler
from infrastructure.file.file_manager import FileManager


@pytest.fixture
def mock_file_manager() -> MagicMock:
    """IFileManager whose sweep succeeds and removes nothing."""
    mock = MagicMock(spec=IFileManager)
    mock.cleanup_old_files.return_value = Result.success(CleanupStats(files_removed=0, bytes_freed=0, errors=()))
    return mock


class TestRunOnce:
    """A single sweep delegates to the file manager and never raises."""

    def test_sweeps_with_configured_max_age(self, mock_file_manager: MagicMock) -> None:
        """The configured age is what the file manager is asked to enforce."""
        FileCleanupScheduler(mock_file_manager, max_file_age_hours=12.5, interval_seconds=60).run_once()

        mock_file_manager.cleanup_old_files.assert_called_once_with(12.5)

    def test_survives_failure_result(self, mock_file_manager: MagicMock) -> None:
        """A failed sweep is logged, not raised."""
        mock_file_manager.cleanup_old_files.return_value = Result.failure(configuration_error("bad age"))

        FileCleanupScheduler(mock_file_manager, max_file_age_hours=1, interval_seconds=60).run_once()

    def test_survives_exception(self, mock_file_manager: MagicMock) -> None:
        """An unexpected exception must not kill the background thread."""
        mock_file_manager.cleanup_old_files.side_effect = OSError("disk gone")

        FileCleanupScheduler(mock_file_manager, max_file_age_hours=1, interval_seconds=60).run_once()


class TestLoop:
    """The background thread sweeps on start and then once per interval."""

    def test_sweeps_immediately_and_repeats(self, mock_file_manager: MagicMock) -> None:
        """Start triggers a sweep right away, then more at each interval, until stopped."""
        two_sweeps = threading.Event()
        calls: list[float] = []

        def sweep(max_age_hours: float) -> Result[CleanupStats]:
            calls.append(max_age_hours)
            if len(calls) >= 2:
                two_sweeps.set()
            return Result.success(CleanupStats(files_removed=0, bytes_freed=0, errors=()))

        mock_file_manager.cleanup_old_files.side_effect = sweep
        scheduler = FileCleanupScheduler(mock_file_manager, max_file_age_hours=3, interval_seconds=0.05)

        scheduler.start()
        try:
            assert two_sweeps.wait(timeout=5), "Scheduler should sweep repeatedly"
        finally:
            scheduler.stop()

        assert scheduler._thread is not None
        assert not scheduler._thread.is_alive()
        assert set(calls) == {3}

    def test_removes_old_files_from_disk(self, tmp_path: Path) -> None:
        """End to end with a real FileManager: aged files disappear, fresh ones stay."""
        manager = FileManager(str(tmp_path / "uploads"), str(tmp_path / "audio"))
        old_audio = Path(manager.output_folder) / "paper_timed.mp3"
        fresh_audio = Path(manager.output_folder) / "other_timed.mp3"
        for path in (old_audio, fresh_audio):
            path.write_bytes(b"x")
        old = time.time() - 2 * 3600
        os.utime(old_audio, (old, old))

        scheduler = FileCleanupScheduler(manager, max_file_age_hours=1, interval_seconds=60)
        scheduler.start()
        try:
            deadline = time.monotonic() + 5
            while old_audio.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            scheduler.stop()

        assert not old_audio.exists()
        assert fresh_audio.exists()


class TestServingProcess:
    """Background services run in exactly the process that serves requests."""

    def test_runs_without_reloader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With debug off there is no reloader and no WERKZEUG_RUN_MAIN."""
        monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
        assert is_serving_process(debug=False)

    def test_skips_reloader_parent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With debug on, the watching parent process must not run them."""
        monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
        assert not is_serving_process(debug=True)

    def test_runs_in_reloader_child(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With debug on, the serving child is marked by WERKZEUG_RUN_MAIN."""
        monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")
        assert is_serving_process(debug=True)
