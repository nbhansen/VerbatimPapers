# domain/text/text_pipeline.py - Text Processing Pipeline with Result[T] pattern
"""Text processing pipeline using Result[T] pattern for error handling.

No exceptions thrown - all errors returned as Result[T].
"""

import logging
import re
from typing import TYPE_CHECKING, Optional
import unicodedata

from ..errors import ApplicationError, ErrorCode, Result
from ..interfaces import ITextPipeline

logger = logging.getLogger(__name__)

# Invisible characters PDF extraction leaves inside words: soft hyphen,
# zero-width space/joiners, word joiner and byte-order mark.
_INVISIBLE_CHARS = re.compile("[\u00ad\u200b-\u200d\u2060\ufeff]")

if TYPE_CHECKING:
    from ..interfaces import ILLMProvider


class TextPipeline(ITextPipeline):
    """Text processing pipeline using Result[T] pattern.

    Pure functions with no exceptions - all errors returned as Result[T].
    """

    def __init__(
        self,
        llm_provider: Optional["ILLMProvider"] = None,
        enable_cleaning: bool = True,
        enable_plain_english: bool = False,
    ):
        self.llm_provider = llm_provider
        self.enable_cleaning = enable_cleaning
        self.enable_plain_english = enable_plain_english

    def _maybe_apply_plain_english(self, text: str) -> str:
        """Apply plain English conversion if enabled, returning original on failure."""
        if not self.enable_plain_english or not self.llm_provider:
            return text
        plain_result = self._apply_plain_english_conversion(text)
        if plain_result.is_success and plain_result.value is not None:
            logger.debug("Plain English applied: %d chars", len(plain_result.value))
            return plain_result.value
        return text

    def clean_text(self, raw_text: str) -> Result[str]:
        """Clean and prepare text for TTS processing.

        Returns Result with cleaned text or error.
        """
        logger.debug("TextPipeline.clean_text(): Input %d chars", len(raw_text))

        if not raw_text:
            return Result.failure(
                ApplicationError(
                    code=ErrorCode.TEXT_EXTRACTION_FAILED, message="Empty text provided for cleaning", retryable=False
                )
            )

        try:
            if not self.enable_cleaning or not self.llm_provider:
                llm_available = self.llm_provider is not None
                logger.debug("Using basic cleanup (cleaning=%s, llm_provider=%s)", self.enable_cleaning, llm_available)
                result = self._basic_text_cleanup(raw_text)
                logger.debug("Basic cleanup result: %d chars", len(result))
                result = self._maybe_apply_plain_english(result)
                return Result.success(result)

            # Use LLM for advanced cleaning
            logger.debug("Using LLM cleaning (provider: %s)", type(self.llm_provider).__name__)
            cleaning_prompt = self._generate_cleaning_prompt(raw_text)
            logger.debug("Generated cleaning prompt (%d chars)", len(cleaning_prompt))

            logger.debug("Calling LLM API...")
            llm_result = self.llm_provider.generate_content(cleaning_prompt)

            if llm_result.is_success:
                cleaned = llm_result.value
                cleaned_len = len(cleaned) if cleaned else 0
                logger.debug("LLM success: %d chars returned", cleaned_len)
                # Basic validation of LLM output
                if cleaned and len(cleaned) > len(raw_text) * 0.05:  # At least 5% of original length
                    logger.debug("LLM output valid, applying basic cleanup")
                    final_result = self._basic_text_cleanup(cleaned)
                    logger.debug("Basic cleanup complete: %d chars", len(final_result))
                    final_result = self._maybe_apply_plain_english(final_result)
                    logger.debug("Final result: %d chars", len(final_result))
                    return Result.success(final_result)
                else:
                    cleaned_len = len(cleaned) if cleaned else 0
                    logger.debug("LLM output too short (%d chars), trying smaller chunks", cleaned_len)
            else:
                logger.warning("LLM failed: %s", llm_result.error)

            # If large chunk failed, try processing in smaller pieces
            if len(raw_text) > 15000:
                logger.debug("Attempting retry with smaller chunks (15K chars each)")
                chunk_result = self._process_in_chunks(raw_text)
                if chunk_result.is_success:
                    return chunk_result

            # Final fallback to basic cleaning if all else fails
            logger.debug("All attempts failed, using basic cleanup")
            fallback_result = self._basic_text_cleanup(raw_text)
            logger.debug("Fallback result: %d chars", len(fallback_result))
            fallback_result = self._maybe_apply_plain_english(fallback_result)
            return Result.success(fallback_result)

        except Exception as e:
            logger.exception("TextPipeline: Exception during cleaning: %s", e)
            # Even on exception, try to return basic cleaned text
            try:
                fallback = self._basic_text_cleanup(raw_text)
                return Result.success(fallback)
            except Exception:
                return Result.from_exception(e, ErrorCode.TEXT_CLEANING_FAILED, retryable=True)

    async def clean_text_async(self, raw_text: str) -> Result[str]:
        """Clean and prepare text for TTS processing asynchronously.

        Returns Result with cleaned text or error.
        """
        if not raw_text:
            return Result.failure(
                ApplicationError(
                    code=ErrorCode.TEXT_EXTRACTION_FAILED, message="Empty text provided for cleaning", retryable=False
                )
            )

        try:
            if not self.enable_cleaning or not self.llm_provider:
                cleaned = self._basic_text_cleanup(raw_text)
                return Result.success(cleaned)

            # Check if async method is available
            if not hasattr(self.llm_provider, "generate_content_async"):
                logger.debug("Async cleaning not available, using sync method")
                return self.clean_text(raw_text)

            # Use async LLM for advanced cleaning with rate limiting
            cleaning_prompt = self._generate_cleaning_prompt(raw_text)
            result = await self.llm_provider.generate_content_async(cleaning_prompt)

            if result.is_success and result.value is not None:
                cleaned = result.value
                # Basic validation of LLM output
                if len(cleaned) > len(raw_text) * 0.3:  # At least 30% of original length
                    final = self._basic_text_cleanup(cleaned)
                    return Result.success(final)

            # Fallback to basic cleaning if LLM fails
            fallback = self._basic_text_cleanup(raw_text)
            return Result.success(fallback)

        except Exception as e:
            logger.exception("Async LLM cleaning failed: %s", e)
            try:
                fallback = self._basic_text_cleanup(raw_text)
                return Result.success(fallback)
            except Exception:
                return Result.from_exception(e, ErrorCode.TEXT_CLEANING_FAILED, retryable=True)

    def split_into_sentences(self, text: str) -> Result[list[str]]:
        """Split text into sentences for individual processing.

        Returns Result with sentence list or error.
        """
        if not text:
            return Result.success([])

        try:
            clean_text = text

            # Handle abbreviations better - don't split on Dr., Mr., etc.
            # Basic sentence splitting with common edge cases
            sentences = re.split(r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bProf\.)(?<=[.!?])\s+(?=[A-Z])", clean_text)

            # Filter out very short sentences and clean up (immutable)
            valid_sentences = [
                sentence.strip() for sentence in sentences if sentence.strip() and len(sentence.strip()) > 10
            ]

            return Result.success(valid_sentences)

        except Exception as e:
            return Result.from_exception(e, ErrorCode.TEXT_EXTRACTION_FAILED, retryable=False)

    def _process_in_chunks(self, raw_text: str) -> Result[str]:
        """Process large text in smaller chunks.

        Pure function returning Result.
        """
        try:
            chunk_size = 15000
            cleaned_parts = []

            for i in range(0, len(raw_text), chunk_size):
                chunk = raw_text[i : i + chunk_size]
                logger.debug("Processing sub-chunk %d (%d chars)", i // chunk_size + 1, len(chunk))

                sub_prompt = self._generate_cleaning_prompt(chunk)
                if self.llm_provider is not None:
                    sub_result = self.llm_provider.generate_content(sub_prompt)

                    if sub_result.is_success and sub_result.value is not None:
                        cleaned_parts.append(sub_result.value)
                        logger.debug("Sub-chunk success: %d chars", len(sub_result.value))
                    else:
                        # Use basic cleanup for failed sub-chunks
                        cleaned_parts.append(self._basic_text_cleanup(chunk))
                        logger.debug("Sub-chunk failed, using basic cleanup")
                else:
                    # No LLM provider available
                    cleaned_parts.append(self._basic_text_cleanup(chunk))
                    logger.debug("No LLM provider, using basic cleanup")

            if cleaned_parts:
                combined_result = " ".join(cleaned_parts)
                logger.debug("Combined sub-chunks: %d chars", len(combined_result))
                final_result = self._basic_text_cleanup(combined_result)

                # Stage 2: Apply plain English conversion if enabled
                if self.enable_plain_english:
                    plain_result = self._apply_plain_english_conversion(final_result)
                    if plain_result.is_success and plain_result.value is not None:
                        final_result = plain_result.value

                return Result.success(final_result)
            else:
                return Result.failure(
                    ApplicationError(
                        code=ErrorCode.TEXT_CLEANING_FAILED, message="Failed to process any chunks", retryable=True
                    )
                )

        except Exception as e:
            return Result.from_exception(e, ErrorCode.TEXT_CLEANING_FAILED, retryable=True)

    def _basic_text_cleanup(self, text: str) -> str:
        """Basic text cleanup without LLM.

        Pure function - no exceptions thrown.
        """
        # Fold PDF presentation forms (ligatures, full-width letters, no-break
        # spaces) into plain characters; accents, dashes and quotes are kept
        text = unicodedata.normalize("NFKC", text)
        text = _INVISIBLE_CHARS.sub("", text)

        # Remove excessive whitespace (also covers form feeds)
        text = re.sub(r"\s+", " ", text)

        # Drop remaining control characters
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")

        # Clean up punctuation
        text = re.sub(r"\.{3,}", "...", text)  # Multiple dots
        text = re.sub(r"-{2,}", "--", text)  # Multiple dashes

        return text.strip()

    def _generate_cleaning_prompt(self, text: str) -> str:
        """Generate LLM prompt for text cleaning.

        Pure function.
        """
        # Speech pacing comes from ordinary punctuation, which the synthesizer honours:
        # a comma buys ~290ms, a semicolon/dash ~325ms, a colon ~360ms, and a period
        # starts a new sentence (where inter-sentence silence is inserted). Repeated or
        # invented punctuation buys nothing - "." and ".." are phonetically identical,
        # and "..." actively erases the sentence boundary. So the instruction is to
        # restore correct punctuation, never to invent decorative pauses.
        punctuation_instruction = """Restore correct punctuation, which PDF extraction often damages:
- Re-join sentences broken across line breaks, and split sentences that were run together
- Restore commas that were dropped: after introductory clauses, around asides, between list items
- Keep the source's own commas, semicolons, colons and dashes - these carry the speech rhythm
- Prefer commas or dashes over parentheses for asides (parentheses are silent when spoken)
- Use ordinary punctuation only. Do NOT add "..." or repeated punctuation for emphasis or pauses"""

        return f"""Clean this text for text-to-speech by removing ONLY:
- Page numbers, headers, footers
- Reference markers like [1], [2], etc.
- Broken hyphenations across lines
- Excessive whitespace

{punctuation_instruction}

Return ONLY the cleaned text, nothing else.

Text:
{text}"""

    def _generate_plain_english_prompt(self, text: str) -> str:
        """Generate LLM prompt for plain English conversion.

        Pure function.
        """
        return f"""Rewrite the following academic text using simpler language.
IMPORTANT: Do NOT summarize or shorten the content. Keep ALL information, just make it easier to understand.

EXAMPLE OF CORRECT REWRITING:
Original: "The utilization of participatory mechanisms facilitates stakeholder engagement
in urban place-making initiatives."
Correct: "Using participatory methods helps stakeholders get involved in urban place-making projects."
WRONG: "People participate in urban planning." (This is too short - it loses important details)

WORD SUBSTITUTION RULES - Replace complex academic words with simpler alternatives:
• "utilize" → "use"
• "facilitate" → "help" or "make possible"
• "implement" → "carry out" or "do"
• "demonstrate" → "show" or "prove"
• "establish" → "show", "find out", or "set up"
• "indicate" → "show" or "suggest"
• "obtain" → "get" or "receive"
• "evaluate" → "test" or "check"
• "ascertain" → "find out"
• "substantial" → "large", "great", or "a lot of"
• "commence" → "start" or "begin"
• "terminate" → "stop" or "end"
• "acquire" → "buy" or "get"
• "accomplish" → "do" or "finish"
• "constitutes" → "makes up" or "forms"

ELIMINATE POMPOUS PHRASES:
• "in the majority of instances" → "most" or "mostly"
• "at this moment in time" → "now" (or edit out)
• "with regard to" → "about" or "for"
• "in order that" → "so that"
• "prior to" → "before"
• "subsequent to" → "after"
• "in excess of" → "more than"
• "in accordance with" → "in line with" or "because of"

REMOVE UNNECESSARY WORDS that add nothing to meaning:
• "actually", "basically", "currently", "existing", "extremely", "obviously", "quite", "really", "very"
• "as a matter of fact", "at the end of the day", "in due course", "the fact of the matter is"

INSTRUCTIONS:
1. Do NOT skip any sentences
2. Replace complex words with simpler ones from the rules above
3. Ensure that sentences are 15-20 words average though some can be longer - use your judgement
4. Convert passive voice to active voice where possible
5. Keep ALL facts, data, examples, and details from the original
6. Do NOT summarize, condense, or remove any content
7. Do NOT add interpretations or explanations not in the original
8. Maintain the exact same structure and order as the original text

CRITICAL: This is a REWRITING task, not a summarizing task.
The output should be approximately the same length as the input, just with simpler words and clearer sentences.

Text to convert:
{text}"""

    def _apply_plain_english_conversion(self, text: str) -> Result[str]:
        """Apply plain English conversion to already-cleaned text.

        Returns Result with converted text or original on failure.
        """
        if not self.llm_provider:
            logger.debug("Plain English: No LLM provider available, skipping conversion")
            return Result.success(text)

        logger.debug("Plain English: Converting %d chars", len(text))

        try:
            # For very large texts, process in chunks
            if len(text) > 15000:
                logger.debug("Plain English: Processing in chunks")
                chunk_size = 15000
                converted_parts = []

                for i in range(0, len(text), chunk_size):
                    chunk = text[i : i + chunk_size]
                    logger.debug("Plain English chunk %d (%d chars)", i // chunk_size + 1, len(chunk))

                    chunk_prompt = self._generate_plain_english_prompt(chunk)
                    chunk_result = self.llm_provider.generate_content(chunk_prompt)

                    if chunk_result.is_success and chunk_result.value:
                        converted_parts.append(chunk_result.value)
                        logger.debug("Plain English chunk success: %d chars", len(chunk_result.value))
                    else:
                        # Use original chunk if conversion fails
                        converted_parts.append(chunk)
                        logger.debug("Plain English chunk failed, using original")

                if converted_parts:
                    combined_result = " ".join(converted_parts)
                    logger.debug("Plain English: Combined chunks: %d chars", len(combined_result))
                    return Result.success(combined_result)
            else:
                # Process whole text
                plain_english_prompt = self._generate_plain_english_prompt(text)
                result = self.llm_provider.generate_content(plain_english_prompt)

                if result.is_success and result.value:
                    # Basic validation - converted text should be reasonable length
                    if len(result.value) > len(text) * 0.3:  # At least 30% of original
                        logger.debug("Plain English: Success: %d chars", len(result.value))
                        return Result.success(result.value)
                    else:
                        logger.debug("Plain English: Result too short (%d chars), using original", len(result.value))
                else:
                    error_msg = result.error if hasattr(result, "error") else "Unknown error"
                    logger.warning("Plain English: LLM failed: %s", error_msg)

            # Fallback to original text if conversion fails
            logger.debug("Plain English: Using original text as fallback")
            return Result.success(text)

        except Exception as e:
            logger.exception("Plain English: Exception during conversion: %s", e)
            return Result.success(text)
