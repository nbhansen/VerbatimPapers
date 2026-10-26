# application/config/system_config.py
"""SystemConfig - Central configuration management using composition.

Configuration Hierarchy
=======================

SystemConfig is the single source of truth for all application settings.
It uses composition of specialized config classes for separation of concerns.

Config Structure (from config.yaml):
------------------------------------
tts:                    # TTSConfig - TTS engine selection and rate limiting
  engine: "piper"       # "piper" (local) or "gemini" (cloud)
  concurrent_requests   # Max parallel TTS requests
  request_delay_seconds # Delay between requests (rate limiting)

piper:                  # PiperConfig - Piper-specific settings
  model_name            # Voice model (e.g., "en_US-lessac-medium")
  download_dir          # Where to store downloaded models
  length_scale          # Speech speed (1.0 = normal)
  noise_scale           # Speech variability
  noise_w               # Pronunciation/timing variability
  sentence_silence      # Seconds of silence between sentences

gemini:                 # GeminiConfig - Gemini API settings (optional)
  api_key               # API key (or set GEMINI_API_KEY env var)
  model_name            # Model to use
  style_prompt          # Delivery/tone directive prefixed to the text

files:                  # FileConfig - File storage paths
  upload_folder         # Where uploaded PDFs go
  audio_folder          # Where generated audio goes
  max_file_size_mb      # Upload size limit

cleanup:                # FileCleanupConfig - Auto-cleanup scheduler
  enabled               # Enable/disable cleanup
  max_age_seconds       # Delete files older than this
  check_interval_seconds # How often to check

text_processing:        # TextProcessingConfig - Text pipeline settings
  enable_cleaning       # Use LLM for text cleaning
  enable_plain_english  # Simplify language
  llm_chunk_size        # Max chars per chunk sent to LLM for cleaning

performance:            # PerformanceConfig - Processing optimization
  enable_async          # Use async TTS processing
  audio_target_chunk_size # Target chars per audio chunk

flask:                  # FlaskConfig - Web server settings
  host, port, debug

ocr:                    # OCRConfig - Tesseract settings
  language, dpi, psm

llm:                    # LLMConfig - LLM provider settings
  model_name, max_tokens, temperature

admin:                  # AdminConfig - /admin endpoint gating
  enabled               # Default false; /admin/* returns 404 when disabled
                        # Optional token via secrets.admin_token (X-Admin-Token header)

Dependencies
------------
- tts.engine determines which of piper/gemini config is active
- text_processing.enable_cleaning requires llm config
- cleanup requires files config (uses same folders)

Environment Variables
---------------------
- GEMINI_API_KEY: API key for Gemini (alternative to config file)
- FLASK_DEBUG: Override flask.debug setting
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import cast

import yaml

from domain.config.tts_config import GeminiConfig, PiperConfig, TTSConfig, TTSEngine

from .app_configs import AdminConfig, FlaskConfig
from .file_configs import FileCleanupConfig, FileConfig
from .logging_config import LoggingConfig
from .processing_configs import LLMConfig, OCRConfig, PerformanceConfig, TextProcessingConfig

logger = logging.getLogger(__name__)

# Type for YAML configuration values
YAMLValue = str | int | float | bool | list[str] | dict[str, object] | None
# Type for config accessor function
ConfigAccessor = Callable[[str, YAMLValue | None], YAMLValue]


@dataclass(frozen=True)
class SystemConfig:
    """Single source of truth for all application configuration using composition."""

    # Core configurations
    tts: TTSConfig
    files: FileConfig
    cleanup: FileCleanupConfig
    text_processing: TextProcessingConfig
    performance: PerformanceConfig
    flask: FlaskConfig
    ocr: OCRConfig
    llm: LLMConfig

    # Engine-specific configs (only one should be set based on tts.engine)
    gemini: GeminiConfig | None = None
    piper: PiperConfig | None = None

    # Logging configuration (default INFO, auto-derived from flask.debug in from_yaml)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)

    # Admin endpoint gating (disabled by default)
    admin: AdminConfig = field(default_factory=AdminConfig)

    # System-level settings
    project_root: str = ""

    def __post_init__(self) -> None:
        """Initialize project root if not set."""
        if not self.project_root:
            secure_root = self._get_secure_project_root()
            object.__setattr__(self, "project_root", secure_root)

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "SystemConfig":
        """Load configuration from YAML file."""
        # Load and validate YAML file
        yaml_config = cls._load_and_validate_yaml(config_path)

        # Create configuration accessor function
        get_config = cls._create_config_accessor(yaml_config)

        # Parse TTS configuration
        tts_config = cls._parse_tts_config(get_config)

        # Always parse both engine configs for backward compatibility
        # The appropriate one will be used based on tts.engine
        gemini_config = cls._parse_gemini_config(get_config)
        piper_config = cls._parse_piper_config(get_config)

        # Parse all other configurations
        file_config = cls._parse_file_config(get_config)
        cleanup_config = cls._parse_cleanup_config(get_config)
        text_config = cls._parse_text_processing_config(get_config)
        performance_config = cls._parse_performance_config(get_config)
        flask_config = cls._parse_flask_config(get_config)
        ocr_config = cls._parse_ocr_config(get_config)
        llm_config = cls._parse_llm_config(get_config)
        logging_config = cls._parse_logging_config(get_config, flask_config)
        admin_config = cls._parse_admin_config(get_config)

        # Get project root
        project_root = str(get_config("system.project_root", ""))

        # Create the composed configuration
        config = cls(
            tts=tts_config,
            files=file_config,
            cleanup=cleanup_config,
            text_processing=text_config,
            performance=performance_config,
            flask=flask_config,
            ocr=ocr_config,
            llm=llm_config,
            gemini=gemini_config,
            piper=piper_config,
            logging_config=logging_config,
            admin=admin_config,
            project_root=project_root,
        )

        config.validate()
        return config

    @classmethod
    def _load_and_validate_yaml(cls, config_path: str) -> dict[str, object]:
        """Load and validate YAML configuration file."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file {config_path} not found. "
                "Please copy config.example.yaml to config.yaml and customize it."
            )

        try:
            with Path(config_file).open() as f:
                yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

        return cast(dict[str, object], yaml_config)

    @classmethod
    def _create_config_accessor(cls, yaml_config: dict[str, object]) -> ConfigAccessor:
        """Create helper function to access nested YAML values."""

        def get_config(yaml_path: str, default: YAMLValue | None = None) -> YAMLValue:
            keys = yaml_path.split(".")
            value: YAMLValue = yaml_config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]  # type: ignore[assignment]
                else:
                    return default
            return value

        return get_config

    @classmethod
    def _parse_tts_config(cls, get_config: ConfigAccessor) -> TTSConfig:
        """Parse TTS configuration."""
        engine_str = get_config("tts.engine", "piper")
        if not engine_str:
            raise ValueError("Missing required configuration: tts.engine")
        engine_str = str(engine_str).strip().lower()
        try:
            engine = TTSEngine(engine_str)
        except ValueError as e:
            valid_engines = [e.value for e in TTSEngine]
            raise ValueError(f"Invalid TTS engine '{engine_str}'. Must be one of: {valid_engines}") from e

        return TTSConfig(
            engine=engine,
            concurrent_requests=cls._parse_int_value(get_config("tts.concurrent_requests", 4), 4, 1, 10),
            request_delay_seconds=cls._parse_float_value(get_config("tts.request_delay_seconds", 2.0), 2.0, 0.1, 10.0),
        )

    @classmethod
    def _parse_gemini_config(cls, get_config: ConfigAccessor) -> GeminiConfig:
        """Parse Gemini-specific configuration."""
        return GeminiConfig(
            api_key=cls._parse_optional_string_value(get_config("secrets.google_ai_api_key", None)),
            voice_name=cls._parse_string_value(get_config("tts.gemini.voice_name", "Kore"), "Kore"),
            model_name=cls._parse_string_value(
                get_config("tts.gemini.model_name", "gemini-2.5-flash"), "gemini-2.5-flash"
            ),
            style_prompt=cls._parse_optional_string_value(get_config("tts.gemini.style_prompt", None)),
            min_request_interval=cls._parse_float_value(
                get_config("tts.gemini.min_request_interval", 2.0), 2.0, 0.1, 10.0
            ),
            max_retries=cls._parse_int_value(get_config("tts.gemini.max_retries", 3), 3, 1, 10),
            base_retry_delay=cls._parse_int_value(get_config("tts.gemini.base_retry_delay", 16), 16, 1, 60),
        )

    @classmethod
    def _parse_piper_config(cls, get_config: ConfigAccessor) -> PiperConfig:
        """Parse Piper-specific configuration."""
        return PiperConfig(
            model_name=cls._parse_string_value(
                get_config("tts.piper.model_name", "en_US-lessac-medium"), "en_US-lessac-medium"
            ),
            length_scale=cls._parse_float_value(get_config("tts.piper.length_scale", 1.0), 1.0, 0.5, 2.0),
            model_repository_url=cls._parse_string_value(
                get_config(
                    "tts.piper.model_repository_url", "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
                ),
                "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0",
            ),
            model_path=cls._parse_optional_string_value(get_config("tts.piper.model_path", None)),
            config_path=cls._parse_optional_string_value(get_config("tts.piper.config_path", None)),
            speaker_id=cls._parse_optional_int_value(get_config("tts.piper.speaker_id", None)),
            noise_scale=cls._parse_float_value(get_config("tts.piper.noise_scale", 0.667), 0.667, 0.0, 2.0),
            noise_w=cls._parse_float_value(get_config("tts.piper.noise_w", 0.8), 0.8, 0.0, 2.0),
            sentence_silence=cls._parse_float_value(get_config("tts.piper.sentence_silence", 0.2), 0.2, 0.0, 5.0),
            download_dir=cls._parse_string_value(get_config("tts.piper.download_dir", "piper_models"), "piper_models"),
            use_gpu=cls._parse_bool_value(get_config("tts.piper.use_gpu", True), True),
        )

    @classmethod
    def _parse_file_config(cls, get_config: ConfigAccessor) -> FileConfig:
        """Parse file handling configuration."""
        # Process allowed extensions
        allowed_ext = get_config("files.allowed_extensions", ["pdf"])
        allowed_extensions: frozenset[str]
        if isinstance(allowed_ext, list):
            allowed_extensions = frozenset(allowed_ext)
        else:
            allowed_extensions = frozenset(ext.strip() for ext in str(allowed_ext).split(","))

        # Process audio extensions
        audio_ext = get_config("files.audio_extensions", ["wav", "mp3"])
        audio_extensions: frozenset[str]
        if isinstance(audio_ext, list):
            audio_extensions = frozenset(audio_ext)
        else:
            audio_extensions = frozenset(ext.strip() for ext in str(audio_ext).split(","))

        return FileConfig(
            upload_folder=cls._parse_string_value(get_config("files.upload_folder", "uploads"), "uploads"),
            audio_folder=cls._parse_string_value(get_config("files.audio_folder", "audio_outputs"), "audio_outputs"),
            max_file_size_mb=cls._parse_int_value(get_config("files.max_file_size_mb", 100), 100, 1, 1000),
            allowed_extensions=allowed_extensions,
            audio_extensions=audio_extensions,
        )

    @classmethod
    def _parse_cleanup_config(cls, get_config: ConfigAccessor) -> FileCleanupConfig:
        """Parse file cleanup configuration."""
        return FileCleanupConfig(
            enabled=cls._parse_bool_value(get_config("files.cleanup.enabled", True), True),
            max_file_age_hours=cls._parse_float_value(
                get_config("files.cleanup.max_file_age_hours", 24.0), 24.0, 0.1, 168.0
            ),
            auto_cleanup_interval_hours=cls._parse_float_value(
                get_config("files.cleanup.auto_cleanup_interval_hours", 6.0), 6.0, 0.1, 24.0
            ),
        )

    @classmethod
    def _parse_text_processing_config(cls, get_config: ConfigAccessor) -> TextProcessingConfig:
        """Parse text processing configuration."""
        return TextProcessingConfig(
            enable_cleaning=cls._parse_bool_value(get_config("text_processing.enable_text_cleaning", True), True),
            enable_plain_english=cls._parse_bool_value(
                get_config("text_processing.enable_plain_english_conversion", False), False
            ),
            llm_chunk_size=cls._parse_int_value(
                get_config("text_processing.llm_chunk_size", 50000), 50000, 10000, 200000
            ),
            audio_target_chunk_size=cls._parse_int_value(
                get_config("text_processing.audio_target_chunk_size", 3000), 3000, 100, 10000
            ),
            audio_max_chunk_size=cls._parse_int_value(
                get_config("text_processing.audio_max_chunk_size", 5000), 5000, 100, 20000
            ),
        )

    @classmethod
    def _parse_performance_config(cls, get_config: ConfigAccessor) -> PerformanceConfig:
        """Parse performance configuration."""
        return PerformanceConfig(
            enable_async_audio=cls._parse_bool_value(get_config("performance.enable_async_audio", True), True),
            audio_concurrent_chunks=cls._parse_int_value(get_config("audio.concurrent_chunks", 4), 4, 1, 20),
            max_concurrent_operations=cls._parse_int_value(
                get_config("performance.max_concurrent_operations", 2), 2, 1, 8
            ),
        )

    @classmethod
    def _parse_flask_config(cls, get_config: ConfigAccessor) -> FlaskConfig:
        """Parse Flask application configuration."""
        return FlaskConfig(
            debug=cls._parse_bool_value(get_config("app.debug", True), True),
            host=cls._parse_string_value(get_config("app.host", "127.0.0.1"), "127.0.0.1"),
            port=cls._parse_int_value(get_config("app.port", 5000), 5000, 1000, 65535),
        )

    @classmethod
    def _parse_admin_config(cls, get_config: ConfigAccessor) -> AdminConfig:
        """Parse admin endpoint gating configuration."""
        return AdminConfig(
            enabled=cls._parse_bool_value(get_config("admin.enabled", False), False),
            token=cls._parse_optional_string_value(get_config("secrets.admin_token", None)),
        )

    @classmethod
    def _parse_logging_config(cls, get_config: ConfigAccessor, flask_config: FlaskConfig) -> LoggingConfig:
        """Parse logging configuration.

        If logging.level is not set, derives from flask.debug:
        debug=true -> DEBUG, debug=false -> INFO.
        """
        explicit_level = get_config("logging.level", None)
        level = ("DEBUG" if flask_config.debug else "INFO") if explicit_level is None else str(explicit_level).upper()

        fmt = cls._parse_string_value(
            get_config("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        json_format = cls._parse_bool_value(get_config("logging.json_format", False), False)

        return LoggingConfig(level=level, format=fmt, json_format=json_format, console_output=True)

    @classmethod
    def _parse_ocr_config(cls, get_config: ConfigAccessor) -> OCRConfig:
        """Parse OCR configuration."""
        return OCRConfig(
            dpi=cls._parse_int_value(get_config("ocr.dpi", 300), 300, 150, 600),
            threshold=cls._parse_int_value(get_config("ocr.threshold", 180), 180, 100, 240),
            language=cls._parse_string_value(get_config("ocr.language", "eng"), "eng"),
        )

    @classmethod
    def _parse_llm_config(cls, get_config: ConfigAccessor) -> LLMConfig:
        """Parse LLM configuration."""
        # Get the API key which may be shared with Gemini
        api_key = cls._parse_optional_string_value(get_config("secrets.google_ai_api_key", None))

        return LLMConfig(
            model_name=cls._parse_optional_string_value(get_config("llm.model_name", None)),
            concurrent_requests=cls._parse_int_value(get_config("llm.concurrent_requests", 3), 3, 1, 10),
            request_delay_seconds=cls._parse_float_value(get_config("llm.request_delay_seconds", 0.5), 0.5, 0.1, 5.0),
            api_key=api_key,
        )

    def validate(self) -> None:
        """Validate configuration and fail fast with clear error messages."""
        # Engine-specific validation
        if self.tts.engine == TTSEngine.GEMINI:
            if self.gemini and not self.gemini.api_key:
                raise ValueError("GOOGLE_AI_API_KEY is required when TTS_ENGINE=gemini")
            if self.gemini and self.gemini.api_key == "YOUR_GOOGLE_AI_API_KEY":
                raise ValueError("Please set a valid GOOGLE_AI_API_KEY (not the placeholder value)")

        if self.tts.engine == TTSEngine.PIPER and self.piper and not self.piper.model_name:
            raise ValueError("PIPER_MODEL_NAME cannot be empty when using Piper TTS")

        # Security validation for all directory paths
        for folder_name, folder_path in [
            ("UPLOAD_FOLDER", self.files.upload_folder),
            ("AUDIO_FOLDER", self.files.audio_folder),
            ("PIPER_DOWNLOAD_DIR", self.piper.download_dir if self.piper else ""),
        ]:
            if folder_path:  # Skip if piper is None
                self._validate_directory_security(folder_name, folder_path, self.project_root)

        # Validate file management settings
        if self.cleanup.enabled:
            if self.cleanup.max_file_age_hours <= 0:
                raise ValueError("MAX_FILE_AGE_HOURS must be positive when file cleanup is enabled")
            if self.cleanup.auto_cleanup_interval_hours <= 0:
                raise ValueError("AUTO_CLEANUP_INTERVAL_HOURS must be positive when file cleanup is enabled")

    @staticmethod
    def _get_secure_project_root() -> str:
        """Get secure project root path with environment override support."""
        env_root = os.environ.get("PROJECT_ROOT")
        if env_root:
            env_path = Path(env_root).resolve()
            if env_path.exists() and env_path.is_dir():
                return str(env_path)
        return str(Path.cwd())

    @staticmethod
    def _validate_directory_security(name: str, path: str, project_root: str) -> None:
        """Validate directory path for security vulnerabilities."""
        if not path or path.isspace():
            raise ValueError(f"{name} cannot be empty or whitespace")

        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = Path(project_root) / path_obj

        path_resolved = path_obj.resolve()
        project_resolved = Path(project_root).resolve()

        try:
            path_resolved.relative_to(project_resolved)
        except ValueError as e:
            raise ValueError(f"{name} must be within project directory: {path}") from e

    # Parsing helper methods
    @staticmethod
    def _parse_string_value(value: YAMLValue, default: str) -> str:
        """Parse string from YAML value."""
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _parse_optional_string_value(value: YAMLValue, default: str | None = None) -> str | None:
        """Parse optional string from YAML value."""
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _parse_bool_value(value: YAMLValue, default: bool | None = None) -> bool:
        """Parse boolean from various representations."""
        if value is None:
            if default is None:
                raise ValueError("Boolean value cannot be None without a default")
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return default if default is not None else False

    @staticmethod
    def _parse_int_value(value: YAMLValue, default: int, min_val: int | None = None, max_val: int | None = None) -> int:
        """Parse integer from various representations with validation."""
        if value is None:
            return default

        try:
            if isinstance(value, bool):
                parsed = 1 if value else 0
            elif isinstance(value, (str, int, float)):
                parsed = int(value)
            else:
                raise ValueError(f"Cannot convert {type(value)} to int")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Value must be a valid integer, got: {value}") from e

        if min_val is not None and parsed < min_val:
            raise ValueError(f"Value must be >= {min_val}, got: {parsed}")
        if max_val is not None and parsed > max_val:
            raise ValueError(f"Value must be <= {max_val}, got: {parsed}")

        return parsed

    @staticmethod
    def _parse_optional_int_value(value: YAMLValue, default: int | None = None) -> int | None:
        """Parse optional integer from YAML value."""
        if value is None:
            return default
        try:
            if isinstance(value, bool):
                return 1 if value else 0
            elif isinstance(value, (str, int, float)):
                return int(value)
            else:
                return default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _parse_float_value(
        value: YAMLValue, default: float, min_val: float | None = None, max_val: float | None = None
    ) -> float:
        """Parse float from various representations with validation."""
        if value is None:
            return default

        try:
            if isinstance(value, (str, int, float)):
                parsed = float(value)
            else:
                raise ValueError(f"Cannot convert {type(value)} to float")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Value must be a valid number, got: {value}") from e

        if min_val is not None and parsed < min_val:
            raise ValueError(f"Value must be >= {min_val}, got: {parsed}")
        if max_val is not None and parsed > max_val:
            raise ValueError(f"Value must be <= {max_val}, got: {parsed}")

        return parsed

    def log_summary(self, log: logging.Logger | None = None) -> None:
        """Log configuration summary."""
        log = log or logger
        lines = [
            "=" * 50,
            "VerbatimPapers - Configuration",
            "=" * 50,
            f"Log Level: {self.logging_config.level}",
            f"TTS Engine: {self.tts.engine.value}",
            f"Text Cleaning: {'Enabled' if self.text_processing.enable_cleaning else 'Disabled'}",
            f"Plain English Conversion: {'Enabled' if self.text_processing.enable_plain_english else 'Disabled'}",
            f"Async Audio: {'Enabled' if self.performance.enable_async_audio else 'Disabled'}",
            f"Audio Concurrent Chunks: {self.performance.audio_concurrent_chunks}",
            f"TTS Concurrent Requests: {self.tts.concurrent_requests}",
            f"LLM Concurrent Requests: {self.llm.concurrent_requests}",
            f"Upload Folder: {self.files.upload_folder}",
            f"Audio Folder: {self.files.audio_folder}",
            f"File Cleanup: {'Enabled' if self.cleanup.enabled else 'Disabled'}",
        ]

        if self.cleanup.enabled:
            lines.extend(
                [
                    f"Max File Age: {self.cleanup.max_file_age_hours} hours",
                    f"Cleanup Interval: {self.cleanup.auto_cleanup_interval_hours} hours",
                ]
            )

        if self.tts.engine == TTSEngine.GEMINI and self.gemini:
            api_key_status = "Set" if self.gemini.api_key else "Missing"
            lines.extend(
                [
                    f"Gemini API Key: {api_key_status}",
                    f"Gemini Voice: {self.gemini.voice_name}",
                ]
            )
        elif self.tts.engine == TTSEngine.PIPER and self.piper:
            lines.extend(
                [
                    f"Piper Model: {self.piper.model_name}",
                    f"Piper Models Dir: {self.piper.download_dir}",
                ]
            )

        lines.append("=" * 50)
        log.info("\n".join(lines))
