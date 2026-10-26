# domain/text/segmentation.py - Sentence splitting and size-bounded chunking
"""The one sentence splitter and the one chunker.

Every place that needs sentences or size-bounded pieces of text uses these:
LLM cleaning chunks, TTS chunks and Piper's per-sentence segments. Piper's
segments become the read-along highlights, so the splitter decides what a
listener sees highlighted.

Pure functions. Original whitespace (including the line breaks that LLM
cleaning uses to spot headers, footers and hyphenation) is kept inside each
sentence and chunk.
"""

import re

# Words that end in a period without ending the sentence, compared lower-case
# and without their final period
_ABBREVIATIONS = frozenset(
    {
        "al",  # et al.
        "approx",
        "ca",
        "cf",
        "dr",
        "ed",
        "eds",
        "eq",
        "eqs",
        "fig",
        "figs",
        "jr",
        "mr",
        "mrs",
        "ms",
        "p",
        "pp",
        "prof",
        "sr",
        "st",
        "vol",
        "vs",
    }
)

# Dotted acronyms and initialisms such as "e.g", "i.e", "U.S" (final period excluded).
# Letters only: "0.5" or "2.1" at the end of a sentence must still end it.
_DOTTED_ACRONYM = re.compile(r"^(?:[^\W\d_]\.)+[^\W\d_]$")

# A single-letter initial such as "R.", alone or at the start of a string
_INITIAL = re.compile(r"^[^\W\d_]\.(?:\s|$)")

# Quotes and brackets that may close a sentence or open the next one
_CLOSERS = "\"')]\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"
_OPENERS = "\"'([\N{LEFT DOUBLE QUOTATION MARK}\N{LEFT SINGLE QUOTATION MARK}"

# Terminal punctuation, any closing quotes or brackets, then whitespace
_CANDIDATE_BREAK = re.compile(f"[.!?]+[{re.escape(_CLOSERS)}]*\\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping each sentence's own punctuation.

    A break needs terminal punctuation, whitespace, and then a capital letter
    (any script), optionally behind an opening quote or bracket. A period after
    a known abbreviation, a dotted acronym or a chain of initials ("J. R. R.") is
    not a break. A decimal at the end of a sentence ("was 0.5.") still ends it.
    """
    return [text[a:b] for a, b in _sentence_spans(text)]


def chunk_text(text: str, max_size: int) -> list[str]:
    """Pack text into chunks of at most max_size characters.

    Whole sentences are packed together. A sentence longer than max_size gets
    chunks of its own, split between words, and a word longer than max_size is
    split between characters. Each chunk is a slice of the original text, so
    nothing is dropped and line breaks inside a chunk survive.

    Raises ValueError for a non-positive max_size, which is a programming error.
    """
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")

    chunks: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None

    def pack(a: int, b: int) -> None:
        nonlocal current
        if current is not None and b - current[0] <= max_size:
            current = (current[0], b)
            return
        if current is not None:
            chunks.append(current)
        current = (a, b)

    def flush() -> None:
        nonlocal current
        if current is not None:
            chunks.append(current)
            current = None

    for a, b in _sentence_spans(text):
        if b - a <= max_size:
            pack(a, b)
            continue
        # Oversized: give it chunks of its own, split between words, then characters
        flush()
        for word in re.finditer(r"\S+", text[a:b]):
            for i in range(word.start(), word.end(), max_size):
                pack(a + i, a + min(i + max_size, word.end()))
        flush()
    flush()
    return [text[a:b] for a, b in chunks]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence in text, with surrounding whitespace excluded."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _CANDIDATE_BREAK.finditer(text):
        if _is_sentence_break(text, match):
            _add_trimmed(spans, text, start, match.start() + len(match.group().rstrip()))
            start = match.end()
    _add_trimmed(spans, text, start, len(text))
    return spans


def _add_trimmed(spans: list[tuple[int, int]], text: str, a: int, b: int) -> None:
    piece = text[a:b]
    if piece.strip():
        a += len(piece) - len(piece.lstrip())
        b -= len(piece) - len(piece.rstrip())
        spans.append((a, b))


def _is_sentence_break(text: str, match: re.Match[str]) -> bool:
    following = text[match.end() :].lstrip(_OPENERS)
    if not following or not following[0].isupper():
        return False

    punctuation = match.group().rstrip()
    if punctuation.rstrip(_CLOSERS) != ".":
        return True  # "!", "?", "..." and combinations always end a sentence here

    before = text[: match.start()].split()
    word = before[-1].lstrip(_OPENERS) if before else ""
    if word.lower() in _ABBREVIATIONS or _DOTTED_ACRONYM.match(word):
        return False
    if len(word) == 1 and word.isupper():
        # Part of a chain of initials ("J. R. R. Tolkien") rather than a sentence
        # ending in a letter ("Appendix B. The ...")
        previous_is_initial = len(before) > 1 and _INITIAL.match(before[-2]) is not None
        return not (previous_is_initial or _INITIAL.match(following))
    return True
