# application/config/processing_configs.py
"""Text processing, OCR, and performance configuration classes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextProcessingConfig:
    """Text processing configuration."""

    enable_cleaning: bool = True
    llm_chunk_size: int = 50000  # Large chunks for LLM text cleaning
    audio_target_chunk_size: int = 3000


@dataclass(frozen=True)
class LLMConfig:
    """LLM configuration for text cleaning.

    model_name is a text model. It is deliberately independent of
    GeminiConfig.model_name, which must be a speech (*-tts) model.
    """

    model_name: str = "gemini-2.5-flash"
    concurrent_requests: int = 3
    request_delay_seconds: float = 0.5


@dataclass(frozen=True)
class OCRConfig:
    """OCR configuration."""

    dpi: int = 300
    language: str = "eng"


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance and resource management configuration."""

    enable_async_audio: bool = True
    audio_concurrent_chunks: int = 4
    max_concurrent_operations: int = 2
