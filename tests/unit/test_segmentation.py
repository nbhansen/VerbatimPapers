"""The one sentence splitter and the one chunker (domain/text/segmentation.py)."""

import pytest

from domain.text.segmentation import chunk_text, split_sentences


class TestSplitSentences:
    """split_sentences decides Piper's segments, and so the read-along highlights."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("One. Two! Three? Four.", ["One.", "Two!", "Three?", "Four."]),
            ("Dr. Smith met Prof. Jones. They talked.", ["Dr. Smith met Prof. Jones.", "They talked."]),
            ("Smith et al. (2020) agree. See Fig. 3 here.", ["Smith et al. (2020) agree.", "See Fig. 3 here."]),
            ("J. R. R. Tolkien wrote it. Then he left.", ["J. R. R. Tolkien wrote it.", "Then he left."]),
            ("In the U.S. Army, e.g. Fort Bragg. Done.", ["In the U.S. Army, e.g. Fort Bragg.", "Done."]),
            ("The value was 3.14 today. Next.", ["The value was 3.14 today.", "Next."]),
            ("Wait... What happened? Nobody knows.", ["Wait...", "What happened?", "Nobody knows."]),
            ("and so on... but then it went on.", ["and so on... but then it went on."]),
            ('He said "Stop." Then he left.', ['He said "Stop."', "Then he left."]),
            ("He said “Stop.” “Why?” she asked.", ["He said “Stop.”", "“Why?” she asked."]),
            ("Results (p < 0.05). More.", ["Results (p < 0.05).", "More."]),
            ("Ørsted came first. Élan followed. Ωmega last.", ["Ørsted came first.", "Élan followed.", "Ωmega last."]),
            ("a lowercase start. after a period stays together", ["a lowercase start. after a period stays together"]),
            # A number, letter or common word at the end of a sentence still ends it
            ("The effect size was 0.5. The next study failed.", ["The effect size was 0.5.", "The next study failed."]),
            ("See Section 2.1. The model is there.", ["See Section 2.1.", "The model is there."]),
            ("Condition B. The next was C.", ["Condition B.", "The next was C."]),
            ("The answer is no. But we tried.", ["The answer is no.", "But we tried."]),
            ("Apples, pears etc. The end.", ["Apples, pears etc.", "The end."]),
            # A lone initial before a name can't be told from a sentence ending in a
            # letter; it splits, which loses no text (only a chain of initials holds)
            ("By J. Smith today.", ["By J.", "Smith today."]),
        ],
    )
    def test_splits(self, text: str, expected: list[str]) -> None:
        """Breaks at real sentence ends only."""
        assert split_sentences(text) == expected

    def test_text_after_the_last_period_is_kept(self) -> None:
        """Trailing words without terminal punctuation are a sentence, not dropped."""
        assert split_sentences("Hello. World") == ["Hello.", "World"]

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_blank_text_has_no_sentences(self, text: str) -> None:
        """Nothing to say yields nothing."""
        assert split_sentences(text) == []


class TestChunkText:
    """chunk_text packs sentences up to a size limit without losing text."""

    TEXT = "Alpha beta. Gamma delta epsilon zeta eta theta. Iota. " + "Supercalifragilistic word."

    @pytest.mark.parametrize("max_size", [5, 12, 20, 60, 1000])
    def test_never_exceeds_max_and_drops_nothing(self, max_size: int) -> None:
        """Every chunk fits, and the chunks rejoin to the original text."""
        chunks = chunk_text(self.TEXT, max_size)

        assert all(0 < len(chunk) <= max_size for chunk in chunks)
        assert " ".join(chunks).replace(" ", "") == self.TEXT.replace(" ", "")

    def test_packs_whole_sentences(self) -> None:
        """Sentences that fit together share a chunk."""
        assert chunk_text("One. Two. Three. Four.", 10) == ["One. Two.", "Three.", "Four."]

    def test_oversized_sentence_gets_its_own_chunks(self) -> None:
        """A sentence split between words doesn't trail onto its neighbours' chunks."""
        chunks = chunk_text("One two three. Four five six seven eight nine. Ten.", 20)

        assert chunks == ["One two three.", "Four five six seven", "eight nine.", "Ten."]

    def test_line_breaks_survive_inside_chunks(self) -> None:
        """LLM cleaning needs the PDF's line structure to spot headers, page numbers and hyphenation."""
        page = "Journal of Things 12\nThe inter-\nnational study found\nthat results vary.\n17"

        assert chunk_text(page, 1000) == [page]
        assert chunk_text(page, 40) == [
            "Journal of Things 12\nThe inter-\nnational",
            "study found\nthat results vary.\n17",
        ]

    def test_split_sentences_keeps_inner_line_breaks(self) -> None:
        """A sentence is a slice of the original text, not a re-spaced copy."""
        assert split_sentences("One\ntwo. Three\nfour.") == ["One\ntwo.", "Three\nfour."]

    def test_blank_text_has_no_chunks(self) -> None:
        """Nothing to chunk yields nothing."""
        assert chunk_text("  ", 10) == []

    def test_rejects_non_positive_size(self) -> None:
        """A zero size is a programming error, not a request for empty chunks."""
        with pytest.raises(ValueError, match="positive"):
            chunk_text("Text.", 0)
