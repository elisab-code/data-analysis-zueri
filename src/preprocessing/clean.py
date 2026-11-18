"""Text cleaning steps shared by both text versions (light and strong).

Each function takes and returns a single string. Missing values (NaN) are
handled by the caller on DataFrame level, not here.
"""

import re
from functools import lru_cache

import ftfy

# Matches line-break tags only: <br/>, <br>, <br />, <BR/>.
# Other text in angle brackets (e.g. "<Schlagloch>") is written by citizens and kept.
LINE_BREAK_TAG = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)

# One or more whitespace characters (space, tab, line break, ...).
WHITESPACE = re.compile(r"\s+")

# local part @ domain . top-level domain, e.g. max.muster@example.ch
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# Punctuation removed together with a phrase ("Guten Tag," / "Danke!").
PHRASE_PUNCTUATION = r"[,.!]*"


def fix_encoding(text: str) -> str:
    """Repair mojibake (UTF-8 bytes that were decoded as Windows-1252).

    Examples: "MÃ¼ll" -> "Müll". Correct text stays unchanged.
    """
    return ftfy.fix_encoding(text)


def remove_html(text: str) -> str:
    """Replace HTML line breaks with a single space.

    The only real tag in `detail` is the line break ("<br/>").
    """
    return LINE_BREAK_TAG.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (spaces, tabs, line breaks) into one space and strip both ends.

    Example: "  Lampe   defekt  " -> "Lampe defekt" (also for tabs and line breaks)
    """
    return WHITESPACE.sub(" ", text).strip()


def mask_emails(text: str, placeholder: str) -> str:
    """Replace e-mail addresses with `placeholder` (personal data, see data review).

    Example: "Von: <max@example.ch>" -> "Von: <EMAIL>" for placeholder "EMAIL".
    Words with "@" that are not addresses (e.g. "@stadtzuerich") are kept.
    """
    return EMAIL.sub(placeholder, text)


def remove_phrases(text: str, phrases: list[str]) -> str:
    """Remove greeting and closing phrases such as "Guten Tag" or "Freundliche Grüsse".

    - case-insensitive: "guten tag" in `phrases` also removes "Guten Tag"
    - whole words only: "danke" must not remove part of "Dankeschön" 
    - punctuation directly after a phrase (",", ".", "!") is removed with it
    Leftover spaces are not collapsed here (see normalize_whitespace).
    """
    if not phrases:
        return text
    return _phrase_pattern(tuple(phrases)).sub(" ", text)


@lru_cache(maxsize=8)
def _phrase_pattern(phrases: tuple[str, ...]) -> re.Pattern:
    """Compile all phrases into one pattern (cached: built once, not for every text)."""
    # Longest first: alternatives are tried from left to right.
    ordered = sorted(phrases, key=len, reverse=True)
    alternatives = "|".join(re.escape(phrase) for phrase in ordered)
    return re.compile(rf"\b(?:{alternatives})\b{PHRASE_PUNCTUATION}", flags=re.IGNORECASE)


def light_clean(text: str, phrases: list[str], email_placeholder: str) -> str:
    """Light cleaning for sentence embeddings: keep word forms and sentence context.

    Order: fix_encoding -> remove_html -> mask_emails -> remove_phrases -> normalize_whitespace
    """
    text = fix_encoding(text)
    text = remove_html(text)
    text = mask_emails(text, email_placeholder)
    text = remove_phrases(text, phrases)
    return normalize_whitespace(text)
