# tests/unit/test_text_pipeline_tdd.py
"""TDD tests for TextPipeline - comprehensive coverage following red-green-refactor cycle.

Tests pure text processing logic without external dependencies.
"""

from collections.abc import Callable
from unittest.mock import Mock

from domain.errors import Result
from domain.interfaces import ITextPipeline
from domain.text.text_pipeline import TextPipeline

# Curly quotes and em dash as escapes so the source stays unambiguous
NON_ASCII_TEXT = "Søren Kierkegaard\u2019s \u201cEither/Or\u201d \u2014 naïve café"


class TestTextPipelineBasicFunctionality:
    """TDD tests for basic TextPipeline functionality."""

    def test_text_pipeline_creation_with_defaults(self):
        """Should create text pipeline with sensible defaults."""
        pipeline = TextPipeline()

        assert pipeline.llm_provider is None
        assert pipeline.enable_cleaning is True

    def test_text_pipeline_creation_with_custom_settings(self):
        """Should create text pipeline with custom configuration."""
        mock_llm = Mock()
        pipeline = TextPipeline(llm_provider=mock_llm, enable_cleaning=False)

        assert pipeline.llm_provider == mock_llm
        assert pipeline.enable_cleaning is False

    def test_text_pipeline_implements_interface(self):
        """Should properly implement ITextPipeline interface."""
        pipeline = TextPipeline()
        assert isinstance(pipeline, ITextPipeline)

        # Check all interface methods are implemented
        assert hasattr(pipeline, "clean_text")


class TestTextCleaningTDD:
    """TDD tests for text cleaning functionality."""

    def test_basic_text_cleanup_removes_excessive_whitespace(self):
        """Should normalize whitespace in text."""
        pipeline = TextPipeline(enable_cleaning=False)  # Disable LLM cleaning

        messy_text = "This  is    text   with\n\n\nexcessive     whitespace."
        result = pipeline.clean_text(messy_text)

        assert result.is_success
        assert result.value == "This is text with excessive whitespace."

    def test_basic_text_cleanup_removes_pdf_artifacts(self):
        """Should remove PDF extraction artifacts without touching real characters."""
        pipeline = TextPipeline(enable_cleaning=False)

        text = "Text with\fform feed, a \ufb01ne liga\u00adture, zero\u200bwidth and\u00a0no-break\x07."
        result = pipeline.clean_text(text)

        assert result.is_success
        assert result.value == "Text with form feed, a fine ligature, zerowidth and no-break."

    def test_basic_text_cleanup_keeps_non_ascii_text(self):
        """Accented letters, em dashes and curly quotes carry meaning and speech rhythm."""
        pipeline = TextPipeline(enable_cleaning=False)

        text = NON_ASCII_TEXT
        result = pipeline.clean_text(text)

        assert result.is_success
        assert result.value == text

    def test_llm_cleaning_keeps_non_ascii_text(self):
        """The cleanup pass over the LLM's output must not strip non-ASCII either."""
        text = NON_ASCII_TEXT
        mock_llm = Mock()
        mock_llm.generate_content.return_value = Result.success(text)

        result = TextPipeline(llm_provider=mock_llm, enable_cleaning=True).clean_text(text)

        mock_llm.generate_content.assert_called_once()
        assert result.is_success
        assert result.value == text

    def test_basic_text_cleanup_normalizes_punctuation(self):
        """Should normalize excessive punctuation."""
        pipeline = TextPipeline(enable_cleaning=False)

        text = "Excessive dots.... and dashes---- in text."
        result = pipeline.clean_text(text)

        assert result.is_success
        assert result.value is not None
        assert "..." in result.value  # Normalized to three dots
        assert "...." not in result.value
        assert "--" in result.value  # Normalized to two dashes
        assert "----" not in result.value

    def test_llm_cleaning_with_successful_provider(self):
        """Should use LLM for cleaning when available and successful."""
        mock_llm = Mock()
        mock_llm.generate_content.return_value = Result.success("Cleaned text with proper formatting.")

        pipeline = TextPipeline(llm_provider=mock_llm, enable_cleaning=True)

        raw_text = "Raw   messy    text needs cleaning."
        result = pipeline.clean_text(raw_text)

        # Should call LLM provider
        mock_llm.generate_content.assert_called_once()

        # Should use LLM result
        assert result.is_success
        assert result.value is not None
        assert "Cleaned text with proper formatting." in result.value

    def test_llm_cleaning_falls_back_on_failure(self):
        """Should fall back to basic cleaning when LLM fails."""
        mock_llm = Mock()
        mock_llm.generate_content.return_value = Result.failure(Mock())

        pipeline = TextPipeline(llm_provider=mock_llm, enable_cleaning=True)

        raw_text = "Raw   messy    text needs cleaning."
        result = pipeline.clean_text(raw_text)

        # Should still return cleaned text using basic method
        assert result.is_success
        assert result.value == "Raw messy text needs cleaning."

    def test_llm_cleaning_falls_back_on_short_response(self):
        """Should fall back when LLM response is too short."""
        mock_llm = Mock()
        mock_llm.generate_content.return_value = Result.success(
            "X"
        )  # Less than 5% of original (1 char vs 75 chars = 1.3%)

        pipeline = TextPipeline(llm_provider=mock_llm, enable_cleaning=True)

        raw_text = "This is a much longer text that should not result in a very short response."
        result = pipeline.clean_text(raw_text)

        # Should fall back to basic cleaning
        assert result.is_success
        assert result.value is not None
        assert len(result.value) > len("X")
        assert "This is a much longer text" in result.value

    def test_llm_cleaning_exception_handling(self):
        """Should handle LLM exceptions gracefully."""
        mock_llm = Mock()
        mock_llm.generate_content.side_effect = Exception("LLM service error")

        pipeline = TextPipeline(llm_provider=mock_llm, enable_cleaning=True)

        raw_text = "Text   with   spacing issues."
        result = pipeline.clean_text(raw_text)

        # Should still return cleaned text
        assert result.is_success
        assert result.value == "Text with spacing issues."

    def test_cleaning_disabled_uses_basic_cleanup(self):
        """Should use basic cleanup when cleaning is disabled."""
        mock_llm = Mock()
        pipeline = TextPipeline(llm_provider=mock_llm, enable_cleaning=False)

        raw_text = "Text   with   spacing."
        result = pipeline.clean_text(raw_text)

        # Should not call LLM
        mock_llm.generate_content.assert_not_called()

        # Should use basic cleanup
        assert result.is_success
        assert result.value == "Text with spacing."


class TestNoPauseMarkupInjectionTDD:
    """The pipeline must not invent punctuation to fake speech pauses.

    Measured against Piper: "." and ".." are phonetically identical, and "..."
    erases the sentence boundary entirely (espeak-ng merges the words), which
    also suppresses the inter-sentence silence. See issue #58.
    """

    def test_cleaning_does_not_double_sentence_punctuation(self):
        """Basic cleanup must leave sentence-ending periods alone."""
        pipeline = TextPipeline(enable_cleaning=False)

        result = pipeline.clean_text("First sentence. Second sentence. Third sentence.")

        assert result.is_success
        assert result.value is not None
        assert result.value == "First sentence. Second sentence. Third sentence."
        assert ".." not in result.value

    def test_cleaning_does_not_inject_ellipses_after_section_headers(self):
        """Section headers must not gain trailing dots - they destroy the boundary."""
        pipeline = TextPipeline(enable_cleaning=False)

        result = pipeline.clean_text("Abstract. This paper presents a method. Introduction. We begin here.")

        assert result.is_success
        assert result.value is not None
        assert "..." not in result.value
        assert "Abstract. This paper" in result.value

    def test_abbreviations_survive_cleaning_intact(self):
        """Trailing dots on abbreviations broke sentence splitting downstream."""
        pipeline = TextPipeline(enable_cleaning=False)

        result = pipeline.clean_text("We follow Smith et al. and cf. Jones for the setup.")

        assert result.is_success
        assert result.value is not None
        assert "et al.." not in result.value
        assert "cf.." not in result.value

    def test_cleaning_does_not_double_commas(self):
        """Doubled commas are a no-op acoustically and corrupt the text."""
        pipeline = TextPipeline(enable_cleaning=False)

        result = pipeline.clean_text("In this paper, we present a method.")

        assert result.is_success
        assert result.value is not None
        assert ",," not in result.value

    def test_ordinary_punctuation_is_preserved(self):
        """Commas, semicolons, colons and dashes carry the rhythm - keep them."""
        pipeline = TextPipeline(enable_cleaning=False)

        text = "We tested three cases: the first, the second; and the third."
        result = pipeline.clean_text(text)

        assert result.is_success
        assert result.value == text


class TestTextPipelineIntegrationTDD:
    """TDD tests for TextPipeline component integration."""

    def test_full_pipeline_processing_workflow(self):
        """Should handle complete text processing workflow."""
        mock_llm = Mock()
        mock_llm.generate_content.return_value = Result.success(
            "This paper presents our algorithm methodology. We propose a new approach."
        )

        pipeline = TextPipeline(llm_provider=mock_llm, enable_cleaning=True)

        raw_text = "Raw   academic   text about algorithms."

        # Clean text
        cleaned = pipeline.clean_text(raw_text)
        assert cleaned.is_success
        assert cleaned.value is not None
        assert "algorithm" in cleaned.value

        # Cleaning must not introduce markup
        assert "<emphasis>" not in cleaned.value

    def test_pipeline_with_all_features_disabled(self):
        """Should work correctly with all enhancement features disabled."""
        pipeline = TextPipeline(enable_cleaning=False)

        text = "Simple   text   with   spacing."

        cleaned = pipeline.clean_text(text)
        assert cleaned.is_success
        assert cleaned.value is not None

        # Basic cleanup only - text is passed through to TTS verbatim otherwise
        assert cleaned.value == "Simple text with spacing."

    def test_pipeline_error_resilience(self):
        """Should be resilient to various error conditions."""
        # Test with None values
        pipeline = TextPipeline()

        # Should handle empty strings gracefully
        clean_empty = pipeline.clean_text("")
        assert clean_empty.is_failure  # Empty text should fail for cleaning


class TestTextPipelinePromptGenerationTDD:
    """TDD tests for LLM prompt generation logic."""

    def test_cleaning_prompt_generation_uses_universal_approach(self):
        """Should use universal academic text processing approach in prompts."""
        pipeline = TextPipeline()

        text = "Sample text for cleaning."
        prompt = pipeline._generate_cleaning_prompt(text)

        assert "text-to-speech" in prompt
        assert "cleaned text" in prompt
        assert "Sample text for cleaning." in prompt

    def test_cleaning_prompt_generation_limits_text_size(self):
        """Should limit text size in prompts to prevent token overflow."""
        pipeline = TextPipeline()

        # Create very long text
        long_text = "Sample text. " * 1000  # Much longer than 5000 chars
        prompt = pipeline._generate_cleaning_prompt(long_text)

        # Should include the text even if long (prompt generation includes full text with template)
        assert "Sample text." in prompt
        assert len(prompt) > 500  # Should include template and some of the text

    def test_cleaning_prompt_generation_includes_instructions(self):
        """Should include proper cleaning instructions."""
        pipeline = TextPipeline()

        text = "Sample text"
        prompt = pipeline._generate_cleaning_prompt(text)

        assert "for text-to-speech" in prompt
        # What the listener doesn't need: page furniture and publisher metadata
        for removed in ("running headers and footers", "DOIs, URLs", "To cite this article", "copyright"):
            assert removed in prompt
        # What must survive: everything that is the paper itself, unaltered
        for kept in ("the title", "abstract", "all body text", "author-year citations"):
            assert kept in prompt
        assert "Do not summarize, shorten, reword or reorder anything." in prompt
        assert "Restore correct punctuation" in prompt
        # Speech rhythm comes from ordinary punctuation; invented pause markup is a
        # no-op at best and erases sentence boundaries at worst.
        assert 'Do NOT add "..."' in prompt


class TestLargeTextRetry:
    """When the LLM fails on a whole chunk, the retry splits it with the shared chunker."""

    def test_retry_splits_at_sentences_not_mid_word(self):
        """Every retried piece is whole sentences, and together they cover the text in order."""
        sentences = [f"Sentence number {i} has a few words in it." for i in range(500)]
        raw_text = " ".join(sentences)
        assert len(raw_text) > 15000

        pieces: list[str] = []

        def generate(prompt: str) -> Result[str]:
            text = prompt.split("Text:\n", 1)[1]
            if len(text) == len(raw_text):
                return Result.failure(Mock())  # The whole chunk fails
            pieces.append(text)
            return Result.success(text)

        mock_llm = Mock()
        mock_llm.generate_content.side_effect = generate

        result = TextPipeline(llm_provider=mock_llm, enable_cleaning=True).clean_text(raw_text)

        assert result.is_success
        assert len(pieces) > 1
        assert all(len(piece) <= 15000 and piece.endswith(".") for piece in pieces)
        assert " ".join(pieces) == raw_text


class TestPlainEnglish:
    """The "Convert to plain English" option rewrites the cleaned text, never losing it."""

    @staticmethod
    def _llm(reply: Callable[[str], Result[str]]) -> Mock:
        llm = Mock()
        llm.generate_content.side_effect = reply
        return llm

    @staticmethod
    def _text_of(prompt: str) -> str:
        return prompt.split("Text to convert:\n", 1)[1]

    def test_rewrite_replaces_the_text(self):
        """With cleaning off, the basic-cleaned text is sent for rewriting and the rewrite is used."""
        llm = self._llm(lambda prompt: Result.success("Using these methods helps people take part."))

        result = TextPipeline(llm_provider=llm, enable_cleaning=False, enable_plain_english=True).clean_text(
            "The   utilization of these mechanisms facilitates participation."
        )

        (prompt,) = [call.args[0] for call in llm.generate_content.call_args_list]
        assert prompt.startswith("Rewrite the following academic text using simpler language.")
        assert self._text_of(prompt) == "The utilization of these mechanisms facilitates participation."
        assert result.value == "Using these methods helps people take part."

    def test_rewrite_follows_llm_cleaning(self):
        """Rewriting starts from the cleaned text, not the raw extraction."""
        replies = iter([Result.success("Cleaned academic sentence here."), Result.success("Simple sentence here.")])
        llm = self._llm(lambda prompt: next(replies))

        result = TextPipeline(llm_provider=llm, enable_cleaning=True, enable_plain_english=True).clean_text(
            "Raw academic sentence [1] here."
        )

        prompts = [call.args[0] for call in llm.generate_content.call_args_list]
        assert self._text_of(prompts[1]) == "Cleaned academic sentence here."
        assert result.value == "Simple sentence here."

    def test_off_by_default(self):
        """Without the option, no rewrite prompt is ever sent."""
        llm = self._llm(lambda prompt: Result.success("Cleaned text that is long enough."))

        TextPipeline(llm_provider=llm, enable_cleaning=True).clean_text("Raw text that is long enough.")

        assert not any("Rewrite the following" in call.args[0] for call in llm.generate_content.call_args_list)

    def test_failed_rewrite_keeps_the_text(self):
        """An LLM error falls back to the unrewritten text rather than failing the conversion."""
        llm = self._llm(lambda prompt: Result.failure(Mock()))

        result = TextPipeline(llm_provider=llm, enable_cleaning=False, enable_plain_english=True).clean_text(
            "Original academic text."
        )

        assert result.value == "Original academic text."

    def test_summary_instead_of_rewrite_is_rejected(self):
        """A reply far shorter than the input is a summary, which would drop content; the original is kept."""
        original = "This sentence carries detail that must survive the rewrite. " * 10
        llm = self._llm(lambda prompt: Result.success("Short summary."))

        result = TextPipeline(llm_provider=llm, enable_cleaning=False, enable_plain_english=True).clean_text(original)

        assert result.value == original.strip()

    def test_long_text_is_rewritten_in_order_and_failed_pieces_kept(self):
        """Over the size limit the text is rewritten piecewise; a failed piece stays as it was."""
        sentences = [f"Sentence {i} is written in a needlessly elaborate academic register." for i in range(700)]
        original = " ".join(sentences)
        calls: list[str] = []

        def reply(prompt: str) -> Result[str]:
            text = self._text_of(prompt)
            calls.append(text)
            if len(calls) == 2:
                return Result.failure(Mock())
            return Result.success(f"<simple {len(calls)}>")

        result = TextPipeline(
            llm_provider=self._llm(reply), enable_cleaning=False, enable_plain_english=True
        ).clean_text(original)

        assert len(calls) > 2
        assert all(len(piece) <= 15000 for piece in calls)
        assert " ".join(calls) == original
        expected = [calls[1] if i == 1 else f"<simple {i + 1}>" for i in range(len(calls))]
        assert result.value == " ".join(expected)
