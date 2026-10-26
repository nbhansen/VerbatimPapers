# tests/unit/test_text_pipeline_tdd.py
"""TDD tests for TextPipeline - comprehensive coverage following red-green-refactor cycle.

Tests pure text processing logic without external dependencies.
"""

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
        assert hasattr(pipeline, "split_into_sentences")


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


class TestSentenceSplittingTDD:
    """TDD tests for sentence splitting functionality."""

    def test_split_into_sentences_basic_splitting(self):
        """Should split text into individual sentences."""
        pipeline = TextPipeline()

        text = "First sentence. Second sentence! Third sentence?"
        result = pipeline.split_into_sentences(text)

        assert result.is_success
        assert result.value is not None
        assert len(result.value) == 3
        assert "First sentence." in result.value
        assert "Second sentence!" in result.value
        assert "Third sentence?" in result.value

    def test_split_into_sentences_filters_short_sentences(self):
        """Should filter out very short sentences."""
        pipeline = TextPipeline()

        text = "Normal sentence here. Yes. Another normal sentence."
        result = pipeline.split_into_sentences(text)

        # Should filter out "Yes." as it's too short
        assert result.is_success
        assert result.value is not None
        assert len(result.value) == 2
        assert "Normal sentence here." in result.value
        assert "Another normal sentence." in result.value
        assert "Yes." not in result.value

    def test_split_into_sentences_handles_abbreviations(self):
        """Should handle common abbreviations correctly."""
        pipeline = TextPipeline()

        text = "Dr. Smith conducted the study. Mr. Jones reviewed it."
        result = pipeline.split_into_sentences(text)

        # Should not split on "Dr." or "Mr."
        assert result.is_success
        assert result.value is not None
        assert len(result.value) == 2
        assert "Dr. Smith conducted the study." in result.value
        assert "Mr. Jones reviewed it." in result.value

    def test_split_into_sentences_handles_natural_text(self):
        """Should handle natural text without markup."""
        pipeline = TextPipeline()

        text = "First sentence with emphasis. Second sentence follows."
        result = pipeline.split_into_sentences(text)

        # Should split naturally on sentence boundaries
        assert result.is_success
        assert result.value is not None
        assert len(result.value) == 2
        assert "First sentence with emphasis." in result.value
        assert "Second sentence follows." in result.value

    def test_split_into_sentences_handles_empty_text(self):
        """Should handle empty or whitespace text."""
        pipeline = TextPipeline()

        empty_result = pipeline.split_into_sentences("")
        assert empty_result.is_success
        assert empty_result.value == []

        whitespace_result = pipeline.split_into_sentences("   ")
        assert whitespace_result.is_success
        assert whitespace_result.value == []

    def test_split_into_sentences_trims_whitespace(self):
        """Should trim whitespace from individual sentences."""
        pipeline = TextPipeline()

        text = "  First sentence.   Second sentence.  "
        result = pipeline.split_into_sentences(text)

        assert result.is_success
        assert result.value is not None
        for sentence in result.value:
            assert sentence == sentence.strip()

    def test_split_into_sentences_handles_numbers_and_decimals(self):
        """Should not split on decimal numbers."""
        pipeline = TextPipeline()

        text = "The value is 3.14159 in this calculation. Next sentence follows."
        result = pipeline.split_into_sentences(text)

        assert result.is_success
        assert result.value is not None
        assert len(result.value) == 2
        assert "3.14159" in result.value[0]


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

        # Split into sentences
        sentences = pipeline.split_into_sentences(cleaned.value)
        assert sentences.is_success
        assert sentences.value is not None
        assert len(sentences.value) >= 1

        # Cleaning must not introduce markup
        all_text = " ".join(sentences.value)
        assert "<emphasis>" not in all_text
        assert "algorithm" in all_text

    def test_pipeline_consistency_across_operations(self):
        """Should maintain text consistency through all operations."""
        pipeline = TextPipeline(enable_cleaning=False)

        original_text = "Test sentence one. Test sentence two."

        # Process through pipeline
        cleaned = pipeline.clean_text(original_text)
        assert cleaned.is_success
        assert cleaned.value is not None
        sentences = pipeline.split_into_sentences(cleaned.value)
        assert sentences.is_success
        assert sentences.value is not None

        # Recombine and strip
        recombined = " ".join(sentences.value)
        final = pipeline.split_into_sentences(recombined)
        assert final.is_success
        assert final.value is not None

        # Should preserve core content verbatim
        assert any("Test sentence one" in sentence for sentence in final.value)
        assert any("Test sentence two" in sentence for sentence in final.value)

    def test_pipeline_handles_large_text_efficiently(self):
        """Should handle larger text blocks without issues."""
        pipeline = TextPipeline(enable_cleaning=False)

        # Create larger text block
        large_text = " ".join([f"Sentence number {i} with content." for i in range(100)])

        sentences = pipeline.split_into_sentences(large_text)
        assert sentences.is_success
        assert sentences.value is not None
        assert len(sentences.value) == 100  # Should split correctly

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

        split_empty = pipeline.split_into_sentences("")
        assert split_empty.is_success
        assert split_empty.value == []

        # Test split returns proper Result type
        sentences = pipeline.split_into_sentences("")
        assert isinstance(sentences, Result)
        assert isinstance(sentences.value, list)
        assert len(sentences.value) == 0


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

        assert "Clean this text for text-to-speech" in prompt
        assert "Page numbers, headers, footers" in prompt
        assert "Restore correct punctuation" in prompt
        # Speech rhythm comes from ordinary punctuation; invented pause markup is a
        # no-op at best and erases sentence boundaries at worst.
        assert 'Do NOT add "..."' in prompt


class TestTextPipelineEdgeCasesTDD:
    """TDD tests for edge cases and error conditions."""

    def test_handles_very_long_sentences(self):
        """Should handle extremely long sentences."""
        pipeline = TextPipeline()

        # Create very long sentence
        long_sentence = "This is a very long sentence " * 100 + "."
        result = pipeline.split_into_sentences(long_sentence)

        # Should still process correctly
        assert result.is_success
        assert result.value is not None
        assert len(result.value) >= 1
        assert len(result.value[0]) > 1000

    def test_handles_text_with_no_punctuation(self):
        """Should handle text without sentence-ending punctuation."""
        pipeline = TextPipeline()

        text = "Text without any punctuation marks just words"
        result = pipeline.split_into_sentences(text)

        # Should return the whole text as one sentence if over minimum length
        assert result.is_success
        assert result.value is not None
        assert len(result.value) == 1
        assert result.value[0] == text

    def test_handles_text_with_only_punctuation(self):
        """Should handle text with only punctuation marks."""
        pipeline = TextPipeline()

        text = "!!! ??? ."  # Shorter punctuation sequence
        result = pipeline.split_into_sentences(text)

        # Should filter out as too short (less than 10 chars)
        assert result.is_success
        assert result.value is not None
        assert len(result.value) == 0
