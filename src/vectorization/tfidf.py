"""TF-IDF vectorization of the strongly cleaned texts (input for NMF)."""

from collections.abc import Iterable

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# The texts are already tokenized by spaCy and joined with spaces, so a token is
# everything between spaces. The default pattern of scikit-learn (\b\w\w+\b) would
# split "e-scooter" and "tempo-30-zone" and drop one-character tokens like "4".
TOKEN_PATTERN = r"\S+"


def build_tfidf(
    texts: Iterable[str],
    ngram_range: tuple[int, int],
    min_df: int | float,
    max_df: int | float,
    exclude_words: Iterable[str] = (),
    exclude_numbers: bool = False,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit a TF-IDF vectorizer and return it with the document-term matrix.

    - ngram_range (1, 2): single words and word pairs ("lampe defekt", "tempo 30")
    - min_df: ignore terms that occur in fewer documents (typos, house numbers)
    - max_df: ignore terms that occur in a larger share of documents
    - exclude_words, exclude_numbers: terms removed after counting (see is_excluded)
    Rows are L2-normalized (scikit-learn default), so the dot product of two
    rows is their cosine similarity.

    With exclusions the vectorizer is fitted twice. The first fit finds the
    terms that pass min_df and max_df. The second fit uses only the kept terms,
    so each row is normalized over the terms that remain.
    """
    texts = list(texts)  # read twice, so a generator would not work
    settings = {
        "token_pattern": TOKEN_PATTERN,
        "lowercase": False,  # already lowercase after strong cleaning
        "ngram_range": tuple(ngram_range),
    }
    vectorizer = TfidfVectorizer(**settings, min_df=min_df, max_df=max_df)
    matrix = vectorizer.fit_transform(texts)

    words = frozenset(exclude_words)
    if not words and not exclude_numbers:
        return vectorizer, matrix.tocsr()

    kept = [
        term
        for term in vectorizer.get_feature_names_out()
        if not is_excluded(term, words, exclude_numbers)
    ]
    vectorizer = TfidfVectorizer(**settings, vocabulary=kept)
    return vectorizer, vectorizer.fit_transform(texts).tocsr()


def is_excluded(term: str, words: frozenset[str], exclude_numbers: bool) -> bool:
    """Whether a term (word or word pair) is removed from the vocabulary.

    Removed: terms that contain one of `words` ("seit", "stehen seit"), and with
    exclude_numbers terms that consist only of numbers ("2", "10 12").
    Kept: "stehen", "tempo 30".
    """
    tokens = term.split(" ")
    if any(token in words for token in tokens):
        return True
    return exclude_numbers and all(is_number(token) for token in tokens)


def is_number(token: str) -> bool:
    """A token without letters: "30", "10.07.2024", "1.5"."""
    return not any(char.isalpha() for char in token)
