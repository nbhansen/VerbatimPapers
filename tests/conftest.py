# tests/conftest.py
"""Fixtures shared across the test suite."""

from collections.abc import Generator
from pathlib import Path
import shutil
import tempfile

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Isolated temporary directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def requires_tesseract() -> None:
    """Skip test if Tesseract OCR is not available."""
    if shutil.which("tesseract") is None:
        pytest.skip("Tesseract OCR not available")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark tests by directory, so `-m unit` and `-m integration` select them."""
    for item in items:
        if "unit/" in str(item.path):
            item.add_marker(pytest.mark.unit)
        elif "integration/" in str(item.path):
            item.add_marker(pytest.mark.integration)
