# tests/infrastructure/file/test_file_manager.py
"""Comprehensive unit tests for FileManager implementation.

Tests initialization, file operations, security controls, and error handling.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from infrastructure.file.file_manager import FileManager


class TestFileManagerInitialization:
    """Test FileManager initialization and directory setup."""

    def test_init_creates_directories_with_absolute_paths(self, temp_dir):
        """Should create directories and convert paths to absolute."""
        upload_path = str(temp_dir / "uploads")
        output_path = str(temp_dir / "outputs")

        manager = FileManager(upload_path, output_path)

        assert manager.upload_folder == str(Path(upload_path).resolve())
        assert manager.output_folder == str(Path(output_path).resolve())
        assert Path(manager.upload_folder).exists()
        assert Path(manager.output_folder).is_absolute()
        assert Path(manager.output_folder).exists()

    def test_init_converts_relative_paths_to_absolute(self, temp_dir):
        """Should convert relative paths to absolute paths."""
        # Change to temp_dir to make relative paths work
        original_cwd = str(Path.cwd())
        try:
            os.chdir(str(temp_dir))

            manager = FileManager("./uploads", "./outputs")

            assert Path(manager.upload_folder).is_absolute()
            assert Path(manager.output_folder).is_absolute()
            assert manager.upload_folder.endswith("uploads")
            assert manager.output_folder.endswith("outputs")
        finally:
            os.chdir(original_cwd)

    def test_init_handles_existing_directories(self, temp_dir):
        """Should work when directories already exist."""
        upload_path = temp_dir / "existing_uploads"
        output_path = temp_dir / "existing_outputs"

        # Pre-create directories
        upload_path.mkdir()
        output_path.mkdir()

        manager = FileManager(str(upload_path), str(output_path))

        assert manager.upload_folder == str(upload_path)
        assert manager.output_folder == str(output_path)
        assert upload_path.exists()
        assert output_path.exists()

    @patch("pathlib.Path.mkdir")
    def test_init_handles_directory_creation_permission_error(self, mock_mkdir, temp_dir):
        """Should raise exception when directory creation fails due to permissions."""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError):
            FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

    def test_init_with_empty_paths_raises_error(self):
        """Should handle empty path strings appropriately."""
        # Empty strings should still work as they resolve to current directory
        manager = FileManager("", "")

        # Should have created directories in current working directory
        assert Path(manager.upload_folder).is_absolute()
        assert Path(manager.output_folder).is_absolute()


class TestFileManagerPathAccessors:
    """Test path accessor methods."""

    def test_get_output_dir_returns_absolute_path(self, temp_dir):
        """Should return absolute path to output directory."""
        output_path = str(temp_dir / "outputs")
        manager = FileManager(str(temp_dir / "uploads"), output_path)

        result = manager.get_output_dir()

        assert result == str(Path(output_path).resolve())
        assert Path(result).is_absolute()

    def test_get_upload_dir_returns_absolute_path(self, temp_dir):
        """Should return absolute path to upload directory."""
        upload_path = str(temp_dir / "uploads")
        manager = FileManager(upload_path, str(temp_dir / "outputs"))

        result = manager.get_upload_dir()

        assert result == str(Path(upload_path).resolve())
        assert Path(result).is_absolute()


class TestFileManagerTempFileOperations:
    """Test temporary file creation and management."""

    def test_save_temp_file_creates_file_with_default_suffix(self, temp_dir):
        """Should create temporary file with default .tmp suffix."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        content = b"test content for temporary file"

        filepath = manager.save_temp_file(content)

        assert filepath.endswith(".tmp")
        assert Path(filepath).exists()
        assert filepath.startswith(manager.output_folder)

        # Verify content
        with Path(filepath).open("rb") as f:
            assert f.read() == content

    def test_save_temp_file_creates_file_with_custom_suffix(self, temp_dir):
        """Should create temporary file with custom suffix."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        content = b"test audio content"

        filepath = manager.save_temp_file(content, suffix=".wav")

        assert filepath.endswith(".wav")
        assert Path(filepath).exists()

        # Verify content
        with Path(filepath).open("rb") as f:
            assert f.read() == content

    def test_save_temp_file_handles_binary_content(self, temp_dir):
        """Should correctly handle binary content."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        binary_content = bytes(range(256))  # All possible byte values

        filepath = manager.save_temp_file(binary_content)

        assert Path(filepath).exists()
        with Path(filepath).open("rb") as f:
            assert f.read() == binary_content

    def test_save_temp_file_handles_empty_content(self, temp_dir):
        """Should handle empty content gracefully."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        filepath = manager.save_temp_file(b"")

        assert Path(filepath).exists()
        assert Path(filepath).stat().st_size == 0

    def test_save_temp_file_handles_large_content(self, temp_dir):
        """Should handle large content without issues."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        large_content = b"x" * (1024 * 1024)  # 1MB content

        filepath = manager.save_temp_file(large_content)

        assert Path(filepath).exists()
        assert Path(filepath).stat().st_size == len(large_content)

    @patch("tempfile.mkstemp")
    def test_save_temp_file_handles_permission_error(self, mock_mkstemp, temp_dir):
        """Should raise appropriate error when temp file creation fails."""
        mock_mkstemp.side_effect = PermissionError("Cannot create temp file")
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        with pytest.raises(PermissionError):
            manager.save_temp_file(b"test content")

    @patch("tempfile.mkstemp")
    @patch("os.fdopen")
    def test_save_temp_file_properly_closes_file_descriptor(self, mock_fdopen, mock_mkstemp, temp_dir):
        """Should properly close file descriptor even if writing fails."""
        mock_fd = MagicMock()
        mock_mkstemp.return_value = (mock_fd, "/fake/path")
        mock_file = MagicMock()
        mock_fdopen.return_value.__enter__.return_value = mock_file
        mock_file.write.side_effect = OSError("Write failed")

        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        with pytest.raises(IOError, match="Write failed"):
            manager.save_temp_file(b"test content")

        # Verify file descriptor context manager was used
        mock_fdopen.assert_called_once_with(mock_fd, "wb")

    @patch("tempfile.mkstemp")
    def test_save_temp_file_uses_correct_directory(self, mock_mkstemp, temp_dir):
        """Should create temp file in output directory."""
        mock_mkstemp.return_value = (1, "/fake/path")
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        with patch("os.fdopen"), patch("builtins.open", mock_open()):
            manager.save_temp_file(b"test")

        mock_mkstemp.assert_called_once_with(suffix=".tmp", dir=manager.output_folder)


class TestFileManagerOutputFileOperations:
    """Test output file creation and management."""

    def test_save_output_file_creates_file_successfully(self, temp_dir):
        """Should create output file with specified content."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        content = b"Final output content"
        filename = "output.wav"

        filepath = manager.save_output_file(content, filename)

        expected_path = str(Path(manager.output_folder) / filename)
        assert filepath == expected_path
        assert Path(filepath).exists()

        with Path(filepath).open("rb") as f:
            assert f.read() == content

    def test_save_output_file_sanitizes_filename_path_traversal(self, temp_dir):
        """Should prevent path traversal attacks in filenames."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        content = b"malicious content"

        # Try various path traversal attempts
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "../../sensitive_file.txt",
        ]

        for malicious_filename in malicious_filenames:
            filepath = manager.save_output_file(content, malicious_filename)

            # Should only use basename, preventing traversal
            expected_basename = Path(malicious_filename).name
            expected_path = str(Path(manager.output_folder) / expected_basename)
            assert filepath == expected_path
            assert filepath.startswith(manager.output_folder)

    def test_save_output_file_raises_error_for_empty_filename(self, temp_dir):
        """Should raise ValueError for empty filename."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        with pytest.raises(ValueError, match="Filename cannot be empty"):
            manager.save_output_file(b"content", "")

    def test_save_output_file_handles_special_characters(self, temp_dir):
        """Should handle filenames with special characters."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        content = b"content with special chars"

        special_filenames = [
            "file with spaces.txt",
            "file_with_underscores.wav",
            "file-with-dashes.mp3",
            "file.with.dots.pdf",
            "file123numbers.txt",
        ]

        for filename in special_filenames:
            filepath = manager.save_output_file(content, filename)

            assert Path(filepath).exists()
            assert filepath.endswith(filename)

    def test_save_output_file_handles_long_filename(self, temp_dir):
        """Should handle very long filenames."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        content = b"content for long filename test"
        long_filename = "very_" * 40 + "long_filename.txt"  # Long filename within OS limits

        filepath = manager.save_output_file(content, long_filename)

        # Should create the file successfully
        assert Path(filepath).exists()
        assert filepath.endswith("long_filename.txt")
        with Path(filepath).open("rb") as f:
            assert f.read() == content

    def test_save_output_file_overwrites_existing_file(self, temp_dir):
        """Should overwrite existing file with same name."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        filename = "overwrite_test.txt"

        # Create first file
        first_content = b"first content"
        filepath1 = manager.save_output_file(first_content, filename)

        # Overwrite with second content
        second_content = b"second content that overwrites"
        filepath2 = manager.save_output_file(second_content, filename)

        assert filepath1 == filepath2
        with Path(filepath2).open("rb") as f:
            assert f.read() == second_content

    @patch("pathlib.Path.write_bytes")
    def test_save_output_file_handles_permission_error(self, mock_write, temp_dir):
        """Should raise appropriate error when file creation fails due to permissions."""
        mock_write.side_effect = PermissionError("Permission denied")
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        with pytest.raises(PermissionError):
            manager.save_output_file(b"content", "test.txt")

    def test_save_output_file_handles_binary_content(self, temp_dir):
        """Should correctly handle binary content in output files."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        binary_content = bytes(range(256))  # All possible byte values

        filepath = manager.save_output_file(binary_content, "binary_test.bin")

        assert Path(filepath).exists()
        with Path(filepath).open("rb") as f:
            assert f.read() == binary_content


class TestFileManagerDeletion:
    """Test file deletion operations and security controls."""

    def test_delete_file_removes_file_in_output_directory(self, temp_dir):
        """Should successfully delete file in output directory."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        # Create a test file
        test_file = Path(manager.output_folder) / "test_delete.txt"
        test_file.write_text("test content")
        assert test_file.exists()

        manager.delete_file(str(test_file))

        assert not test_file.exists()

    def test_delete_file_removes_file_in_upload_directory(self, temp_dir):
        """Should successfully delete file in upload directory."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        # Create a test file
        test_file = Path(manager.upload_folder) / "test_upload.pdf"
        test_file.write_text("test pdf content")
        assert test_file.exists()

        manager.delete_file(str(test_file))

        assert not test_file.exists()

    def test_delete_file_prevents_deletion_outside_managed_directories(self, temp_dir):
        """Should prevent deletion of files outside managed directories."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        # Create file outside managed directories
        outside_file = temp_dir / "outside_file.txt"
        outside_file.write_text("outside content")

        with pytest.raises(ValueError, match="Cannot delete file outside managed directories"):
            manager.delete_file(str(outside_file))

        # File should still exist
        assert outside_file.exists()

    def test_delete_file_rejects_sibling_directory_sharing_prefix(self, temp_dir):
        """A sibling like outputs_evil/ shares the string prefix but is not managed."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        sibling = temp_dir / "outputs_evil"
        sibling.mkdir()
        victim = sibling / "victim.txt"
        victim.write_text("keep me")

        with pytest.raises(ValueError, match="Cannot delete file outside managed directories"):
            manager.delete_file(str(victim))

        assert victim.exists()

    def test_delete_file_prevents_path_traversal_attacks(self, temp_dir):
        """Should prevent path traversal attacks in file deletion."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        # Create a sensitive file outside managed directories
        sensitive_file = temp_dir / "sensitive.txt"
        sensitive_file.write_text("sensitive data")

        # Try various path traversal attempts
        traversal_attempts = [
            "../sensitive.txt",
            f"../../{sensitive_file.name}",
            f"../../../{sensitive_file}",
            f"..\\..\\{sensitive_file.name}",  # Windows style
        ]

        for attempt in traversal_attempts:
            with pytest.raises(ValueError, match="Cannot delete file outside managed directories"):
                manager.delete_file(attempt)

        # Sensitive file should still exist
        assert sensitive_file.exists()

    def test_delete_file_handles_nonexistent_file_gracefully(self, temp_dir):
        """Should not raise error when trying to delete non-existent file in managed directories."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        # Try to delete non-existent file in output directory
        nonexistent_path = str(Path(manager.output_folder) / "nonexistent.txt")

        # Should not raise an error
        manager.delete_file(nonexistent_path)

    def test_delete_file_uses_absolute_path_resolution(self, temp_dir):
        """Should properly resolve relative paths to absolute for security checks."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        # Create a file in output directory
        test_file = Path(manager.output_folder) / "relative_test.txt"
        test_file.write_text("test content")

        # Change directory and use relative path
        original_cwd = str(Path.cwd())
        try:
            os.chdir(manager.output_folder)

            # Should work with relative path when it resolves to managed directory
            manager.delete_file("relative_test.txt")
            assert not test_file.exists()

        finally:
            os.chdir(original_cwd)

    @patch("pathlib.Path.unlink")
    def test_delete_file_handles_permission_error_during_deletion(self, mock_unlink, temp_dir):
        """Should handle permission errors during file deletion."""
        mock_unlink.side_effect = PermissionError("Permission denied")
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        # Create test file path within managed directory
        test_path = str(Path(manager.output_folder) / "test.txt")

        # We need exists to return True for unlink to be called
        with patch("pathlib.Path.exists", return_value=True), pytest.raises(PermissionError):
            manager.delete_file(test_path)


class TestFileManagerErrorHandling:
    """Test error handling and edge cases."""

    @patch("pathlib.Path.mkdir")
    def test_init_handles_os_error_during_directory_creation(self, mock_mkdir, temp_dir):
        """Should handle OS errors during directory creation."""
        mock_mkdir.side_effect = OSError("Disk full")

        with pytest.raises(OSError, match="Disk full"):
            FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

    def test_save_temp_file_with_none_content_raises_error(self, temp_dir):
        """Should handle None content appropriately."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        with pytest.raises(TypeError):
            manager.save_temp_file(None)  # type: ignore

    def test_save_output_file_with_none_content_raises_error(self, temp_dir):
        """Should handle None content appropriately."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        with pytest.raises(TypeError):
            manager.save_output_file(None, "test.txt")  # type: ignore

    def test_save_output_file_with_none_filename_raises_error(self, temp_dir):
        """Should handle None filename appropriately."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        with pytest.raises((TypeError, ValueError)):
            manager.save_output_file(b"content", None)  # type: ignore

    @patch("pathlib.Path.write_bytes")
    def test_save_output_file_handles_disk_full_error(self, mock_write, temp_dir):
        """Should handle disk full errors during file writing."""
        mock_write.side_effect = OSError("No space left on device")

        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        with pytest.raises(OSError, match="No space left on device"):
            manager.save_output_file(b"content", "test.txt")


class TestFileManagerSecurityCompliance:
    """Test security compliance and boundary enforcement."""

    def test_all_paths_are_absolute_after_initialization(self, temp_dir):
        """Should ensure all internal paths are absolute for security."""
        upload_rel = "uploads"
        output_rel = "outputs"

        manager = FileManager(upload_rel, output_rel)

        assert Path(manager.upload_folder).is_absolute()
        assert Path(manager.output_folder).is_absolute()

    def test_directory_boundary_enforcement_comprehensive(self, temp_dir):
        """Should comprehensively test directory boundary enforcement."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        # Create files at various locations
        test_files = {
            "inside_output": Path(manager.output_folder) / "inside.txt",
            "inside_upload": Path(manager.upload_folder) / "inside.txt",
            "outside_sibling": temp_dir / "sibling.txt",
            "outside_parent": temp_dir.parent / "parent.txt",
        }

        for location, file_path in test_files.items():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f"content for {location}")

        # Should allow deletion inside managed directories
        manager.delete_file(str(test_files["inside_output"]))
        manager.delete_file(str(test_files["inside_upload"]))

        assert not test_files["inside_output"].exists()
        assert not test_files["inside_upload"].exists()

        # Should prevent deletion outside managed directories
        for location in ["outside_sibling", "outside_parent"]:
            with pytest.raises(ValueError, match="Cannot delete file outside managed directories"):
                manager.delete_file(str(test_files[location]))

            # Files should still exist
            assert test_files[location].exists()

    def test_filename_sanitization_edge_cases(self, temp_dir):
        """Should handle edge cases in filename sanitization."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))
        content = b"test content"

        edge_case_filenames = [
            "normal_file.txt",
            "file with spaces.txt",
            "/absolute/path/file.txt",  # Should use only basename
            "\\windows\\path\\file.txt",  # Should use only basename
            "~/home/user/file.txt",  # Should use only basename
            "../sibling/file.txt",  # Should use only basename
            "..\\sibling\\file.txt",  # Should use only basename
        ]

        for filename in edge_case_filenames:
            filepath = manager.save_output_file(content, filename)

            # Should always be in output directory
            assert filepath.startswith(manager.output_folder)
            # Should only use the basename
            expected_basename = Path(filename).name
            assert filepath.endswith(expected_basename)
            assert Path(filepath).exists()


class TestFileManagerIntegration:
    """Integration tests for FileManager operations."""

    def test_complete_file_lifecycle(self, temp_dir):
        """Should handle complete file lifecycle: create temp, create output, delete."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        # 1. Create temporary file
        temp_content = b"temporary audio data"
        temp_path = manager.save_temp_file(temp_content, suffix=".wav")
        assert Path(temp_path).exists()

        # 2. Create output file
        final_content = b"final processed audio data"
        output_path = manager.save_output_file(final_content, "final_audio.wav")
        assert Path(output_path).exists()

        # 3. Delete temporary file
        manager.delete_file(temp_path)
        assert not Path(temp_path).exists()
        assert Path(output_path).exists()  # Output should remain

        # 4. Delete output file
        manager.delete_file(output_path)
        assert not Path(output_path).exists()

    def test_concurrent_file_operations_simulation(self, temp_dir):
        """Should handle multiple file operations without conflicts."""
        manager = FileManager(str(temp_dir / "uploads"), str(temp_dir / "outputs"))

        # Simulate concurrent operations
        files_created = []

        for i in range(10):
            # Create temp file
            temp_content = f"temp content {i}".encode()
            temp_path = manager.save_temp_file(temp_content, suffix=f"_{i}.tmp")
            files_created.append(temp_path)

            # Create output file
            output_content = f"output content {i}".encode()
            output_path = manager.save_output_file(output_content, f"output_{i}.txt")
            files_created.append(output_path)

        # Verify all files exist
        for file_path in files_created:
            assert Path(file_path).exists()

        # Clean up all files
        for file_path in files_created:
            manager.delete_file(file_path)
            assert not Path(file_path).exists()


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
