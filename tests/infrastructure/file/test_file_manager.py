# tests/infrastructure/file/test_file_manager.py
"""FileManager: the output and upload folders, and the age-based sweep of both."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from infrastructure.file.file_manager import FileManager


class TestFolders:
    """Both folders exist and are absolute once the manager is built."""

    def test_creates_missing_folders(self, temp_dir: Path) -> None:
        """A fresh install has neither folder; the manager creates them."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        assert Path(manager.upload_folder).is_dir()
        assert Path(manager.get_output_dir()).is_dir()

    def test_relative_paths_become_absolute(self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config paths are relative to where the app starts; the audio engine needs them absolute."""
        monkeypatch.chdir(temp_dir)

        manager = FileManager("uploads", "outputs")

        assert manager.get_output_dir() == str(temp_dir.resolve() / "outputs")
        assert manager.upload_folder == str(temp_dir.resolve() / "uploads")


class TestCleanupOldFiles:
    """Test age-based cleanup of the output and upload directories."""

    @staticmethod
    def _age_file(path: Path, hours: float) -> None:
        """Backdate a file's mtime by the given number of hours."""
        old_time = path.stat().st_mtime - hours * 3600
        os.utime(str(path), (old_time, old_time))

    def test_removes_files_older_than_max_age(self, temp_dir):
        """Should delete old files and spare recent ones."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        old_file = Path(manager.output_folder) / "old.wav"
        old_file.write_bytes(b"x" * 10)
        self._age_file(old_file, hours=2)
        new_file = Path(manager.output_folder) / "new.wav"
        new_file.write_bytes(b"y" * 20)

        result = manager.cleanup_old_files(max_age_hours=1.0)

        assert result.is_success
        assert result.value is not None
        assert result.value.files_removed == 1
        assert result.value.bytes_freed == 10
        assert result.value.errors == ()
        assert not old_file.exists()
        assert new_file.exists()

    def test_also_sweeps_orphaned_uploads(self, temp_dir):
        """Uploads left behind by a crashed job are removed by the same age rule."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        orphan = Path(manager.upload_folder) / "op_paper.pdf"
        orphan.write_bytes(b"%PDF")
        self._age_file(orphan, hours=2)
        in_flight = Path(manager.upload_folder) / "op2_paper.pdf"
        in_flight.write_bytes(b"%PDF")

        result = manager.cleanup_old_files(max_age_hours=1.0)

        assert result.is_success
        assert result.value is not None
        assert result.value.files_removed == 1
        assert not orphan.exists()
        assert in_flight.exists()

    def test_ignores_subdirectories(self, temp_dir):
        """Should never touch directories inside the output folder."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        subdir = Path(manager.output_folder) / "nested"
        subdir.mkdir()
        self._age_file(subdir, hours=48)

        result = manager.cleanup_old_files(max_age_hours=1.0)

        assert result.is_success
        assert result.value is not None
        assert result.value.files_removed == 0
        assert subdir.exists()

    @pytest.mark.parametrize("bad_age", [0.0, -1.0, -24.0, float("nan"), float("inf")])
    def test_rejects_non_positive_or_non_finite_age(self, temp_dir, bad_age):
        """A negative age would put the cutoff in the future and delete everything."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        victim = Path(manager.output_folder) / "current.wav"
        victim.write_bytes(b"audio")

        result = manager.cleanup_old_files(max_age_hours=bad_age)

        assert result.is_failure
        assert victim.exists()

    def test_records_error_and_continues_on_delete_failure(self, temp_dir):
        """A file that fails to delete should be reported, not abort the pass."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        stubborn = Path(manager.output_folder) / "stubborn.wav"
        stubborn.write_bytes(b"x")
        self._age_file(stubborn, hours=2)
        removable = Path(manager.output_folder) / "removable.wav"
        removable.write_bytes(b"y")
        self._age_file(removable, hours=2)

        original_unlink = Path.unlink

        def failing_unlink(self: Path, *args: object, **kwargs: object) -> None:
            if self.name == "stubborn.wav":
                raise OSError("permission denied")
            original_unlink(self, *args, **kwargs)  # type: ignore[arg-type]

        with patch.object(Path, "unlink", failing_unlink):
            result = manager.cleanup_old_files(max_age_hours=1.0)

        assert result.is_success
        assert result.value is not None
        assert result.value.files_removed == 1
        assert len(result.value.errors) == 1
        assert "stubborn.wav" in result.value.errors[0]
        assert not removable.exists()
