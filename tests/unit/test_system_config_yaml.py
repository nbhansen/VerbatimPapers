# tests/unit/test_system_config_yaml.py
from pathlib import Path
from typing import Any

import pytest
import yaml

from application.config.system_config import SystemConfig
from domain.config.tts_config import TTSEngine


def _write_yaml_config(tmp_path: Path, config_dict: dict[str, Any]) -> str:
    """Helper to create a temporary YAML config file under tmp_path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_dict))
    return str(config_file)


class TestSystemConfigYAMLLoading:
    """Test YAML configuration loading functionality."""

    def test_from_yaml_with_minimal_config(self, tmp_path: Path):
        """Test loading minimal YAML configuration."""
        config_data = {"tts": {"engine": "piper"}}

        config_file = _write_yaml_config(tmp_path, config_data)
        config = SystemConfig.from_yaml(config_file)
        assert config.tts.engine == TTSEngine.PIPER
        # Check defaults are applied
        assert config.text_processing.enable_cleaning is True
        assert config.files.max_file_size_mb == 100

    def test_from_yaml_with_complete_config(self, tmp_path: Path):
        """Test loading complete YAML configuration."""
        config_data = {
            "tts": {
                "engine": "gemini",
                "request_delay_seconds": 1.5,
                "gemini": {
                    "model_name": "test-model",
                    "voice_name": "Aoede",
                },
                "piper": {"model_name": "en_US-test-high", "models_dir": "test_models", "length_scale": 1.2},
            },
            "secrets": {"google_ai_api_key": "test-api-key-123"},
            "text_processing": {"enable_text_cleaning": False},
            "files": {
                "upload_folder": "test_uploads",
                "audio_folder": "test_audio",
                "max_file_size_mb": 50,
                "cleanup": {"enabled": False, "max_file_age_hours": 48.0},
            },
        }

        config_file = _write_yaml_config(tmp_path, config_data)
        config = SystemConfig.from_yaml(config_file)

        # TTS settings
        assert config.tts.engine == TTSEngine.GEMINI
        assert config.gemini is not None
        assert config.gemini.model_name == "test-model"
        assert config.gemini.voice_name == "Aoede"
        assert config.tts.request_delay_seconds == 1.5
        assert config.gemini.api_key == "test-api-key-123"

        # Text processing
        assert config.text_processing.enable_cleaning is False

        # File settings
        assert config.files.upload_folder == "test_uploads"
        assert config.files.audio_folder == "test_audio"
        assert config.files.max_file_size_mb == 50
        assert config.cleanup.enabled is False
        assert config.cleanup.max_file_age_hours == 48.0

    def test_from_yaml_max_concurrent_operations(self, tmp_path: Path):
        """Should parse the background-processing concurrency cap."""
        config_data = {"tts": {"engine": "piper"}, "performance": {"max_concurrent_operations": 4}}

        config_file = _write_yaml_config(tmp_path, config_data)
        config = SystemConfig.from_yaml(config_file)
        assert config.performance.max_concurrent_operations == 4

    def test_from_yaml_max_concurrent_operations_default(self, tmp_path: Path):
        """Should default the concurrency cap to 2."""
        config_file = _write_yaml_config(tmp_path, {"tts": {"engine": "piper"}})
        config = SystemConfig.from_yaml(config_file)
        assert config.performance.max_concurrent_operations == 2

    def test_from_yaml_max_concurrent_operations_out_of_range(self, tmp_path: Path):
        """Should reject values outside the 1-8 range."""
        config_data = {"tts": {"engine": "piper"}, "performance": {"max_concurrent_operations": 99}}

        config_file = _write_yaml_config(tmp_path, config_data)
        with pytest.raises(ValueError, match="<= 8"):
            SystemConfig.from_yaml(config_file)

    def test_from_yaml_admin_disabled_by_default(self, tmp_path: Path):
        """Admin endpoints should be opt-in."""
        config_file = _write_yaml_config(tmp_path, {"tts": {"engine": "piper"}})
        config = SystemConfig.from_yaml(config_file)
        assert config.admin.enabled is False
        assert config.admin.token is None

    def test_from_yaml_flask_debug_off_by_default(self, tmp_path: Path):
        """The Werkzeug debugger allows code execution, so it must be opt-in."""
        config_file = _write_yaml_config(tmp_path, {"tts": {"engine": "piper"}})
        config = SystemConfig.from_yaml(config_file)
        assert config.flask.debug is False

    def test_from_yaml_admin_enabled_with_token(self, tmp_path: Path):
        """Should parse admin.enabled and the token from the secrets section."""
        config_data = {
            "tts": {"engine": "piper"},
            "admin": {"enabled": True},
            "secrets": {"admin_token": "s3cret-token"},
        }

        config_file = _write_yaml_config(tmp_path, config_data)
        config = SystemConfig.from_yaml(config_file)
        assert config.admin.enabled is True
        assert config.admin.token == "s3cret-token"  # noqa: S105

    def test_from_yaml_missing_file_raises_error(self):
        """Test that missing YAML file raises appropriate error."""
        with pytest.raises(FileNotFoundError) as exc_info:
            SystemConfig.from_yaml("non_existent_file.yaml")
        assert "config.example.yaml" in str(exc_info.value)

    def test_from_yaml_invalid_yaml_raises_error(self, tmp_path: Path):
        """Test that invalid YAML syntax raises appropriate error."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: syntax: [[[")

        with pytest.raises(ValueError, match="Invalid YAML"):
            SystemConfig.from_yaml(str(config_file))

    def test_from_yaml_type_conversions(self, tmp_path: Path):
        """Test that various type conversions work correctly."""
        config_data = {
            "tts": {"engine": "piper", "piper": {"length_scale": "1.5"}},  # String float
            "text_processing": {
                "enable_text_cleaning": "true",  # String boolean
                "audio_target_chunk_size": 2500,
            },
            "files": {
                "max_file_size_mb": "30",  # String integer
                "cleanup": {
                    "enabled": "yes",  # String boolean
                    "max_file_age_hours": "36.5",  # String float
                    "auto_cleanup_interval_hours": 8,  # Integer as float
                },
            },
        }

        config_file = _write_yaml_config(tmp_path, config_data)
        config = SystemConfig.from_yaml(config_file)

        # Boolean conversions
        assert config.text_processing.enable_cleaning is True
        assert config.cleanup.enabled is True

        # Integer conversions
        assert config.text_processing.audio_target_chunk_size == 2500
        assert isinstance(config.text_processing.audio_target_chunk_size, int)
        assert config.files.max_file_size_mb == 30
        assert isinstance(config.files.max_file_size_mb, int)

        # Float conversions
        assert config.cleanup.max_file_age_hours == 36.5
        assert isinstance(config.cleanup.max_file_age_hours, float)
        assert config.cleanup.auto_cleanup_interval_hours == 8.0
        assert isinstance(config.cleanup.auto_cleanup_interval_hours, float)
        assert config.piper is not None
        assert config.piper.length_scale == 1.5
        assert isinstance(config.piper.length_scale, float)

    def test_from_yaml_boolean_variations(self, tmp_path: Path):
        """Test various boolean representations."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("Yes", True),
            ("on", True),
            ("1", True),
            (1, True),
            (True, True),
            ("false", False),
            ("False", False),
            ("no", False),
            ("off", False),
            ("0", False),
            (0, False),
            (False, False),
            ("invalid", False),  # Invalid strings default to False
        ]

        for i, (value, expected) in enumerate(test_cases):
            config_data = {"tts": {"engine": "piper"}, "text_processing": {"enable_text_cleaning": value}}

            config_file = tmp_path / f"config_{i}.yaml"
            config_file.write_text(yaml.dump(config_data))
            config = SystemConfig.from_yaml(str(config_file))
            assert config.text_processing.enable_cleaning is expected, f"Failed for value: {value}"

    def test_from_yaml_range_validation(self, tmp_path: Path):
        """Test that numeric range validation works."""
        # Test value too low
        config_data = {"tts": {"engine": "piper"}, "files": {"max_file_size_mb": 0}}  # Min is 1

        config_file = _write_yaml_config(tmp_path, config_data)
        with pytest.raises(ValueError, match=">= 1"):
            SystemConfig.from_yaml(config_file)

        # Test value too high
        config_data = {"tts": {"engine": "piper"}, "files": {"max_file_size_mb": 2000}}  # Max is 1000

        config_file_high = tmp_path / "config_high.yaml"
        config_file_high.write_text(yaml.dump(config_data))
        with pytest.raises(ValueError, match="<= 1000"):
            SystemConfig.from_yaml(str(config_file_high))

    def test_from_yaml_missing_required_fields(self, tmp_path: Path):
        """Test that missing required fields use defaults."""
        # Missing TTS engine should use default 'piper'
        config_data: dict[str, Any] = {}

        config_file = _write_yaml_config(tmp_path, config_data)
        config = SystemConfig.from_yaml(config_file)
        # Should use default value
        assert config.tts.engine == TTSEngine.PIPER

    def test_from_yaml_invalid_tts_engine(self, tmp_path: Path):
        """Test that invalid TTS engine raises error."""
        config_data = {"tts": {"engine": "invalid_engine"}}

        config_file = _write_yaml_config(tmp_path, config_data)
        with pytest.raises(ValueError, match=r"Invalid TTS engine.*piper.*gemini"):
            SystemConfig.from_yaml(config_file)

    def test_from_yaml_gemini_validation(self, tmp_path: Path):
        """Test Gemini-specific validation."""
        # Gemini without API key should fail validation
        config_data = {
            "tts": {"engine": "gemini"}
            # No secrets.google_ai_api_key provided
        }

        config_file = _write_yaml_config(tmp_path, config_data)
        with pytest.raises(ValueError, match="GOOGLE_AI_API_KEY is required"):
            SystemConfig.from_yaml(config_file)

    def test_from_yaml_with_null_values(self, tmp_path: Path):
        """Test handling of null/None values in YAML."""
        config_data = {
            "tts": {"engine": "piper"},
            "text_processing": {
                "enable_text_cleaning": None,  # Should use default
            },
        }

        config_file = _write_yaml_config(tmp_path, config_data)
        config = SystemConfig.from_yaml(config_file)
        assert config.text_processing.enable_cleaning is True  # Default value
        assert config.text_processing.llm_chunk_size == 50000  # Default value

    def test_from_yaml_case_insensitive_tts_engine(self, tmp_path: Path):
        """Test that TTS engine is case-insensitive."""
        test_cases = ["PIPER", "Piper", "piper", "PiPeR"]

        for i, engine_value in enumerate(test_cases):
            config_data = {"tts": {"engine": engine_value}}

            config_file = tmp_path / f"config_{i}.yaml"
            config_file.write_text(yaml.dump(config_data))
            config = SystemConfig.from_yaml(str(config_file))
            assert config.tts.engine == TTSEngine.PIPER

    def test_from_yaml_with_example_file(self):
        """Test loading the actual example YAML file if it exists."""
        example_file = Path("config.example.yaml")
        if example_file.exists():
            config = SystemConfig.from_yaml(str(example_file))
            # Verify it loads without errors and has expected defaults
            assert config.tts.engine == TTSEngine.PIPER
            assert isinstance(config.text_processing.enable_cleaning, bool)
            assert isinstance(config.files.max_file_size_mb, int)
            assert isinstance(config.tts.request_delay_seconds, float)
