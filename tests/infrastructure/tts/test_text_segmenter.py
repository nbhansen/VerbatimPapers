"""Comprehensive tests for TextSegmenter.

Tests text processing functionality that is shared across all TTS engines.
These tests validate sentence splitting, duration calculation, text chunking,
and text cleaning for TTS compatibility.
"""

import pytest

from infrastructure.tts.text_segmenter import TextSegmenter


@pytest.fixture
def default_segmenter() -> TextSegmenter:
    """Default TextSegmenter with standard settings."""
    return TextSegmenter()


class TestTextSegmenterSentenceSplitting:
    """Test sentence splitting functionality."""

    def test_split_into_sentences_basic(self, default_segmenter: TextSegmenter) -> None:
        """Should split basic sentences correctly."""
        segmenter = default_segmenter

        text = "This is the first sentence. This is the second sentence! Is this the third sentence?"
        sentences = segmenter.split_into_sentences(text)

        expected = [
            "This is the first sentence.",
            "This is the second sentence!",
            "Is this the third sentence?",
        ]
        assert sentences == expected

    def test_split_into_sentences_handles_abbreviations(self, default_segmenter: TextSegmenter) -> None:
        """Should not split on common abbreviations."""
        segmenter = default_segmenter

        text = "Dr. Smith met with Mr. Johnson. They discussed Prof. Brown's research."
        sentences = segmenter.split_into_sentences(text)

        expected = [
            "Dr. Smith met with Mr. Johnson.",
            "They discussed Prof. Brown's research.",
        ]
        assert sentences == expected

    def test_split_into_sentences_handles_multiple_abbreviations(self, default_segmenter: TextSegmenter) -> None:
        """Should handle multiple abbreviations in one sentence."""
        segmenter = default_segmenter

        text = "Mr. and Mrs. Smith visited Dr. Johnson Sr. yesterday."
        sentences = segmenter.split_into_sentences(text)

        assert len(sentences) == 1
        assert sentences[0] == "Mr. and Mrs. Smith visited Dr. Johnson Sr. yesterday."

    def test_split_into_sentences_empty_input(self, default_segmenter: TextSegmenter) -> None:
        """Should handle empty input gracefully."""
        segmenter = default_segmenter

        assert segmenter.split_into_sentences("") == []
        assert segmenter.split_into_sentences("   ") == []
        assert segmenter.split_into_sentences("\n\t") == []

    def test_split_into_sentences_single_sentence(self, default_segmenter: TextSegmenter) -> None:
        """Should handle single sentence without ending punctuation."""
        segmenter = default_segmenter

        text = "This is a single sentence without ending punctuation"
        sentences = segmenter.split_into_sentences(text)

        assert len(sentences) == 1
        assert sentences[0] == text

    def test_split_into_sentences_cleans_whitespace(self, default_segmenter: TextSegmenter) -> None:
        """Should clean whitespace in split sentences."""
        segmenter = default_segmenter

        text = "  First sentence.   Second sentence with extra spaces.  "
        sentences = segmenter.split_into_sentences(text)

        expected = [
            "First sentence.",
            "Second sentence with extra spaces.",
        ]
        assert sentences == expected

    def test_split_into_sentences_handles_edge_cases(self, default_segmenter: TextSegmenter) -> None:
        """Should handle various edge cases in sentence splitting."""
        segmenter = default_segmenter

        # Numbers and decimals
        text = "The price was $19.99. The temperature was 98.6 degrees."
        sentences = segmenter.split_into_sentences(text)
        assert len(sentences) == 2

        # Ellipsis
        text = "He said... Then he paused. Finally he continued."
        sentences = segmenter.split_into_sentences(text)
        assert len(sentences) >= 2  # Should not split on ellipsis inappropriately

    def test_split_into_sentences_preserves_context(self, default_segmenter: TextSegmenter) -> None:
        """Should preserve sentence context and content."""
        segmenter = default_segmenter

        text = "Chapter 1. Introduction to Machine Learning. Chapter 2. Data Processing."
        sentences = segmenter.split_into_sentences(text)

        # Should handle this appropriately
        assert len(sentences) >= 1
        for sentence in sentences:
            assert sentence.strip()  # No empty sentences
