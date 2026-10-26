# domain/config/tts_config.py - Unified TTS Configuration
from dataclasses import dataclass
from enum import Enum


class TTSEngine(Enum):
    """Enumeration of supported TTS engines."""

    PIPER = "piper"
    GEMINI = "gemini"


# Characters per Gemini TTS request: about a paragraph, so highlights stay readable
DEFAULT_GEMINI_CHUNK_SIZE = 1000


@dataclass(frozen=True)
class GeminiConfig:
    """Configuration for Gemini TTS engine."""

    api_key: str | None = None
    voice_name: str = "Kore"
    model_name: str = "gemini-2.5-flash-preview-tts"  # Speech output needs a *-tts model
    style_prompt: str | None = None
    # Gemini gives no sentence boundaries, so each chunk is one read-along highlight
    chunk_size: int = DEFAULT_GEMINI_CHUNK_SIZE


@dataclass(frozen=True)
class PiperConfig:
    """Configuration for Piper TTS engine."""

    model_name: str = "en_US-lessac-medium"
    length_scale: float = 1.0  # Speed: 1.0=normal, <1.0=faster, >1.0=slower
    model_repository_url: str = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    model_path: str | None = None
    config_path: str | None = None
    speaker_id: int | None = None
    noise_scale: float = 0.667  # Speech variability
    noise_w: float = 0.8  # Pronunciation variability
    download_dir: str = "piper_models"


@dataclass(frozen=True)
class TTSConfig:
    """Base TTS configuration applicable to all engines."""

    engine: TTSEngine
    concurrent_requests: int = 4
    request_delay_seconds: float = 2.0
