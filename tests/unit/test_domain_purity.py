"""Regression test: the domain layer must not depend on anything outside itself.

The architecture docs claim domain/ has no external dependencies; this test
enforces it, so violations like the old direct pdfplumber import (issue #36)
cannot creep back in.
"""

import ast
from pathlib import Path
import sys

import pytest

DOMAIN_DIR = Path("domain")

# Stdlib modules that reach outside the process; external tools belong behind
# interfaces in infrastructure/ (ffmpeg moved out in #75)
FORBIDDEN_STDLIB = {"subprocess"}


def _module_root(module_name: str) -> str:
    return module_name.split(".")[0]


@pytest.mark.unit
class TestDomainPurity:
    """Ensure hexagonal architecture boundaries are respected in domain/."""

    def test_domain_imports_only_stdlib_and_domain(self) -> None:
        """Every import in domain/ must be stdlib (minus process-spawning modules) or domain-internal."""
        allowed_roots = (set(sys.stdlib_module_names) - FORBIDDEN_STDLIB) | {"domain"}
        violations: list[str] = []

        for py_file in sorted(DOMAIN_DIR.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _module_root(alias.name) not in allowed_roots:
                            violations.append(f"{py_file}:{node.lineno}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:  # Relative imports stay within domain/
                        continue
                    if _module_root(node.module or "") not in allowed_roots:
                        violations.append(f"{py_file}:{node.lineno}: from {node.module} import ...")

        assert violations == [], "External imports found in domain/:\n" + "\n".join(violations)
