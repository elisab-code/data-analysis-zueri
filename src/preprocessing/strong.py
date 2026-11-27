"""Strong cleaning for TF-IDF: lowercase lemmas without stopwords and punctuation.

Input are texts that already went through light_clean (encoding, HTML,
e-mails, phrases, whitespace). Numbers are kept on purpose: they carry
context such as "Tempo 30", tram lines or how long a problem has existed.
"""

from collections.abc import Iterable

import spacy
from spacy.language import Language
from spacy.tokens import Token


def load_nlp(model: str) -> Language:
    """Load the spaCy model with only the components needed for lemmas.

    The parser (sentence structure) and the named entity recognizer are not
    needed and would make processing several times slower.
    """
    return spacy.load(model, disable=["parser", "ner"])


def keep_token(token: Token, keep_words: frozenset[str]) -> bool:
    """Decide whether a token is kept in the strongly cleaned text.

    Kept: words and numbers, including compounds with hyphens ("E-Scooter",
    "Tempo-30-Zone") and dates ("10.07.2024").
    Removed: punctuation, whitespace, pure symbols and stopwords, except
    stopwords whose lowercase lemma is in `keep_words` (e.g. "nicht", "seit").
    """
    if token.is_punct or token.is_space:
        return False
    # is_alpha would be too strict: it is False for "E-Scooter" or "Tempo-30-Zone".
    if not any(char.isalnum() for char in token.text):
        return False  # pure symbols such as "+" or "->"
    if token.is_stop and token.lemma_.lower() not in keep_words:
        return False
    return True


def strong_clean(
    texts: Iterable[str],
    nlp: Language,
    keep_words: Iterable[str],
    batch_size: int,
    lemma_fixes: dict[str, str] | None = None,
) -> list[str]:
    """Return one string of lowercase lemmas per input text, in the same order.

    Example: "Die Lampen leuchten nicht." -> "lampe leuchten nicht"
    (with "nicht" in keep_words). Lemmas are normalized with normalize_lemma.
    `lemma_fixes` maps wrong lemmas to the correct form (after ß -> ss).

    nlp.pipe processes the texts in batches, which is much faster than
    calling nlp(text) for every text in a loop.
    """
    keep = frozenset(word.lower() for word in keep_words)
    fixes = lemma_fixes or {}
    return [
        " ".join(
            normalize_lemma(lemma_or_text(token), fixes) for token in doc if keep_token(token, keep)
        )
        for doc in nlp.pipe(texts, batch_size=batch_size)
    ]


def lemma_or_text(token: Token) -> str:
    """The lemma of the token, or its text if the lemma has no letter or digit.

    The German model gives many English words the lemma "--" (e.g. "the", "is").
    Such lemmas would merge unrelated words into one term.
    """
    if any(char.isalnum() for char in token.lemma_):
        return token.lemma_
    return token.text


def normalize_lemma(lemma: str, lemma_fixes: dict[str, str]) -> str:
    """Lowercase, Swiss spelling and manual corrections for known lemmatizer errors.

    The model is trained on German newspaper texts: it turns Swiss "ss" into "ß"
    ("Strasse" -> "straß") and does not know some Swiss words ("Velos" -> "velos").
    """
    lemma = lemma.lower().replace("ß", "ss")
    return lemma_fixes.get(lemma, lemma)
