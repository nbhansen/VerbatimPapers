# infrastructure/tts/text_segmenter.py
"""Shared sentence segmentation for TTS engines."""

import re


class TextSegmenter:
    """Sentence segmentation shared by the TTS engines."""

    def split_into_sentences(self, text: str) -> list[str]:
        """Smart sentence splitting preserving context.

        Handles common abbreviations and edge cases that shouldn't trigger splits.
        """
        if not text.strip():
            return []

        # Don't split on common abbreviations
        text = re.sub(r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.\s*", r"\g<0>@@@", text)

        # Split on sentence boundaries (period, exclamation, question mark followed by space and capital)
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Restore abbreviations
        sentences = [s.replace("@@@", "") for s in sentences]

        # Clean and filter
        return [s.strip() for s in sentences if s.strip()]
