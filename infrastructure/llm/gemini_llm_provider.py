# infrastructure/llm/gemini_llm_provider.py
"""Gemini LLM provider implementation for text processing and content generation.

Uses the unified Google Gen AI SDK for language model operations.
"""

import logging
import time

from google import genai
from google.genai import types

from domain.errors import Result, llm_provider_error
from domain.interfaces import ILLMProvider
from infrastructure.gemini_http import default_http_options

logger = logging.getLogger(__name__)


class GeminiLLMProvider(ILLMProvider):
    """Gemini LLM provider for text processing and content generation."""

    def __init__(self, api_key: str, model_name: str, http_options: types.HttpOptions | None = None):
        self.api_key = api_key
        self.model_name = model_name
        self.http_options = http_options or default_http_options()
        self._initialization_error: str | None = None  # Deferred error handling
        self.client = self._init_client()

    def _init_client(self) -> genai.Client | None:
        """Initialize the Gemini client with API key validation."""
        if not self.api_key or self.api_key == "YOUR_GOOGLE_AI_API_KEY":
            self._initialization_error = "Gemini API key not configured"
            return None

        try:
            return genai.Client(api_key=self.api_key, http_options=self.http_options)
        except Exception as e:
            self._initialization_error = f"Failed to initialize Gemini LLM client: {e}"
            logger.error(self._initialization_error)
            return None

    def generate_content(self, prompt: str) -> Result[str]:
        """Generate content based on a prompt."""
        # Check for deferred initialization errors
        if self._initialization_error:
            logger.error("Gemini LLM initialization failed: %s", self._initialization_error)
            return Result.failure(llm_provider_error(self._initialization_error))

        if not self.client:
            return Result.failure(llm_provider_error("Client not available"))

        # Prompts are document text, so only their length is ever logged
        logger.debug("LLM call: model=%s, %d-char prompt", self.model_name, len(prompt))

        start_time = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=30000,  # Increased from 8192 to avoid empty response bug
                    temperature=0.3,
                ),
            )
        except Exception as e:
            logger.error("LLM API error after %.2fs: %s", time.time() - start_time, str(e)[:200])
            return Result.failure(llm_provider_error(f"Content generation failed: {e!s}"))
        logger.debug("LLM call returned in %.2fs", time.time() - start_time)

        candidate = response.candidates[0] if response.candidates else None
        finish_reason = candidate.finish_reason if candidate else None
        # Text cut off at the token limit would silently drop the rest of the passage
        if finish_reason == types.FinishReason.MAX_TOKENS:
            logger.warning("LLM output hit the token limit for a %d-char prompt; discarding it", len(prompt))
            return Result.failure(llm_provider_error("LLM output was truncated at the token limit"))

        # The SDK joins every text part and skips thought parts; taking only the
        # first part would drop the rest of a reply split across parts
        text = response.text
        if not text:
            logger.warning("LLM returned no text for a %d-char prompt (finish reason: %s)", len(prompt), finish_reason)
            return Result.failure(llm_provider_error("Empty response from LLM"))
        logger.debug("LLM returned %d chars", len(text))
        return Result.success(text)
