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

        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info("LLM API Call: Model=%s, prompt='%s' (%s chars)", self.model_name, prompt_preview, len(prompt))

        try:
            start_time = time.time()

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=30000,  # Increased from 8192 to avoid empty response bug
                    temperature=0.3,
                ),
            )

            api_time = time.time() - start_time
            logger.info("LLM API Success: %.2fs for '%s'", api_time, prompt_preview)

            # Inspect response structure for debugging
            if response:
                try:
                    # Check if response has candidates
                    if hasattr(response, "candidates") and response.candidates:
                        candidate = response.candidates[0]
                        finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
                        logger.debug("Response finish_reason: %s", finish_reason)

                        # Try to get text content
                        if (
                            hasattr(candidate, "content")
                            and candidate.content is not None
                            and hasattr(candidate.content, "parts")
                        ):
                            parts = candidate.content.parts
                            if parts and hasattr(parts[0], "text"):
                                text_content = parts[0].text
                                if text_content:
                                    logger.debug("LLM Response: %s chars returned", len(text_content))
                                    return Result.success(text_content)
                except Exception as e:
                    logger.warning("Error inspecting response structure: %s", e)

            # Try the simple .text accessor as fallback
            if response and hasattr(response, "text") and response.text:
                logger.debug("LLM Response: %s chars returned (via .text accessor)", len(response.text))
                return Result.success(response.text)
            else:
                logger.warning("LLM API: No text in response for '%s'", prompt_preview)
                if response:
                    logger.debug("Response object type: %s", type(response))
                    logger.debug("Response attributes: %s...", dir(response)[:10])  # First 10 attributes
                return Result.failure(llm_provider_error("Empty response from LLM"))

        except Exception as e:
            api_time = time.time() - start_time
            error_str = str(e)
            logger.error("LLM API Error (%.2fs): %s for '%s'", api_time, error_str[:200], prompt_preview)
            return Result.failure(llm_provider_error(f"Content generation failed: {e!s}"))
