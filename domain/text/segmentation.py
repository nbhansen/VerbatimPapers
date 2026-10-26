# domain/text/segmentation.py - Sentence splitting and size-bounded chunking
"""The one sentence splitter and the one chunker.

Every place that needs sentences or size-bounded pieces of text uses these:
LLM cleaning chunks, TTS chunks and Piper's per-sentence segments. Piper's
segments become the read-along highlights, so the splitter decides what a
listener sees highlighted.

Pure functions, no exceptions.
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
        "etc",
        "fig",
        "figs",
        "jr",
        "mr",
        "mrs",
        "ms",
        "no",
        "nos",
        "p",
        "pp",
        "prof",
        "sr",
        "st",
        "vol",
        "vs",
    }
)

# Dotted acronyms and initialisms such as "e.g", "i.e", "U.S" (final period excluded)
_DOTTED_ACRONYM = re.compile(r"^(?:\w\.)+\w$")

# Quotes and brackets that may close a sentence or open the next one
_CLOSERS = "\"')]\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"
_OPENERS = "\"'([\N{LEFT DOUBLE QUOTATION MARK}\N{LEFT SINGLE QUOTATION MARK}"

# Terminal punctuation, any closing quotes or brackets, then whitespace
_CANDIDATE_BREAK = re.compile(f"[.!?]+[{re.escape(_CLOSERS)}]*\\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping each sentence's own punctuation.

    A break needs terminal punctuation, whitespace, and then a capital letter
    (any script), optionally behind an opening quote or bracket. A period after
    a known abbreviation, a single initial or a dotted acronym is not a break.
    Decimals never break because a break needs whitespace after the period.
    """
    sentences: list[str] = []
    start = 0
    for match in _CANDIDATE_BREAK.finditer(text):
        if _is_sentence_break(text, match):
            sentence = text[start : match.start()] + match.group().rstrip()
            if sentence.strip():
                sentences.append(sentence.strip())
            start = match.end()
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return sentences


def chunk_text(text: str, max_size: int) -> list[str]:
    """Pack text into chunks of at most max_size characters.

    Whole sentences are packed together. A sentence longer than max_size gets
    chunks of its own, split between words, and a word longer than max_size is
    split between characters. Nothing is dropped: joining the chunks with spaces
    gives back the text with its whitespace normalized.
    """
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")

    chunks: list[str] = []
    for sentence in split_sentences(" ".join(text.split())):
        if len(sentence) <= max_size:
            _pack(chunks, sentence, max_size)
            continue
        # Oversized: start fresh so its pieces don't trail onto the previous sentence
        chunks.append("")
        for word in sentence.split():
            for i in range(0, len(word), max_size):
                _pack(chunks, word[i : i + max_size], max_size)
        chunks.append("")
    return [chunk for chunk in chunks if chunk]


def _pack(chunks: list[str], piece: str, max_size: int) -> None:
    """Append piece to the last chunk if it fits, otherwise start a new chunk."""
    if chunks and chunks[-1] and len(chunks[-1]) + 1 + len(piece) <= max_size:
        chunks[-1] = f"{chunks[-1]} {piece}"
    elif chunks and not chunks[-1]:
        chunks[-1] = piece
    else:
        chunks.append(piece)


def _is_sentence_break(text: str, match: re.Match[str]) -> bool:
    following = text[match.end() :].lstrip(_OPENERS)
    if not following or not following[0].isupper():
        return False

    punctuation = match.group().rstrip()
    if punctuation.rstrip(_CLOSERS) != ".":
        return True  # "!", "?", "..." and combinations always end a sentence here

    word = text[: match.start()].rsplit(maxsplit=1)[-1] if text[: match.start()].strip() else ""
    word = word.lstrip(_OPENERS)
    if word.lower() in _ABBREVIATIONS:
        return False
    if len(word) == 1 and word.isupper():
        return False  # An initial: "J. Smith"
    return not _DOTTED_ACRONYM.match(word)
