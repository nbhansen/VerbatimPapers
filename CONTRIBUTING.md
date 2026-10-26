# Contributing to VerbatimPapers

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the Issues section
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the bug
   - Expected behavior
   - Actual behavior
   - Screenshots if applicable
   - System information (OS, Python version, etc.)

### Suggesting Features

1. Check if the feature has already been suggested in the Issues section
2. If not, create a new issue with:
   - A clear, descriptive title
   - Detailed description of the feature
   - Use cases and benefits
   - Any implementation ideas you might have

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature/fix
3. Make your changes
4. Run tests and ensure they pass
5. Update documentation if necessary
6. Submit a pull request

### Development Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd VerbatimPapers
   ```

2. Install system dependencies:
   ```bash
   # Fedora
   sudo dnf install tesseract ffmpeg espeak-ng

   # Ubuntu/Debian
   sudo apt install tesseract-ocr ffmpeg espeak-ng

   # Arch
   sudo pacman -S tesseract ffmpeg espeak-ng
   ```

3. Install [uv](https://docs.astral.sh/uv/) (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

4. Install dependencies and pre-commit hooks:
   ```bash
   make dev-setup   # runs: uv sync --extra dev && uv run pre-commit install
   ```

5. Copy and configure the application:
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml with your settings (API keys, preferences)
   ```

6. Run tests to verify setup:
   ```bash
   make test-unit   # runs: uv run python -m pytest tests/unit/
   ```

### Code Style

- Line length: 120 characters
- Strict mypy type checking — use type hints on all functions
- Google-style docstrings
- Use `TYPE_CHECKING` guards for circular imports
- First-party import order: `application`, `domain`, `infrastructure`

Run all checks:
```bash
make check       # pre-commit (lint + format + typecheck) + unit tests
make lint        # uv run ruff check . --fix
make format      # uv run ruff format .
make typecheck   # uv run mypy .
```

### Architecture Guidelines

This project uses **hexagonal architecture** — dependencies always point inward toward the domain layer.

- **`domain/`** — Pure business logic with no external dependencies. Contains interfaces, models, errors, and core engines.
- **`infrastructure/`** — Implements interfaces defined in `domain/interfaces.py`. Contains TTS providers, OCR, LLM, and file management.
- **`application/`** — Orchestrates domain + infrastructure. Contains configuration, DI context, and service coordination.

Key patterns:
- **`Result[T]`** (`domain/errors.py`) — Return `Result.success(value)` or `Result.failure(ApplicationError(...))` instead of raising exceptions
- **Deferred error handling** — TTS provider constructors never raise; initialization errors are stored and returned as `Result.failure()` on first use
- **Immutable DI** — `ServiceContainer` (`application/container/service_container.py`, implementing `IServiceContainer` from `domain/container/`) uses lazy initialization with factory lambdas
- **Interfaces** — Define clear interfaces in `domain/interfaces.py` for all external dependencies

### Testing

The project uses pytest with markers for test categorization:

1. **Run the test suite**:
   ```bash
   uv run python -m pytest                       # All tests (or: make test)
   uv run python -m pytest tests/unit/           # Unit tests only (or: make test-unit)
   uv run python -m pytest tests/integration/    # Integration tests (or: make test-integration)
   uv run python -m pytest -k "test_name"        # Specific test
   uv run python -m pytest -m unit               # By marker
   ```

2. **Test Categories** (markers defined in `pyproject.toml`):
   - `unit` — Fast tests with no external dependencies
   - `integration` — Tests with mocked external services
   - `external` — Tests requiring real external services (manual)
   - `slow` — Tests taking more than 5 seconds

3. **Test Requirements**:
   - New domain logic should have unit tests
   - New infrastructure implementations should verify interface contracts
   - Integration tests should verify end-to-end workflows

### Documentation

- Update README.md if architectural changes are made
- Add docstrings to new functions and classes following Google style

### Adding New Features

#### Adding a New TTS Provider
1. Create `infrastructure/tts/your_provider.py` implementing `ITTSEngine` from `domain/interfaces.py`
2. Add configuration dataclass to `domain/config/tts_config.py`
3. Add factory branch in `ServiceContainer._create_tts_engine()`
4. Write unit tests and integration tests

#### Text Processing Features
1. Define interface in `domain/interfaces.py`
2. Implement in `domain/text/`
3. Add infrastructure provider if needed
4. Wire through `ServiceContainer`
5. Write tests

#### Audio Processing Features
1. Define interface if needed
2. Implement in `domain/audio/`
3. Wire through `ServiceContainer`
4. Test with both TTS engines

## Pull Request Process

1. Ensure all tests pass (`make test`)
2. Ensure code style checks pass (`make check`)
3. Update documentation if needed
4. The PR will be merged once you have the sign-off of at least one maintainer

## Questions?

Feel free to open an issue for any questions about contributing.
