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
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit a TF-IDF vectorizer and return it with the document-term matrix.

    - ngram_range (1, 2): single words and word pairs ("nicht mehr", "tempo 30")
    - min_df: ignore terms that occur in fewer documents (typos, house numbers)
    - max_df: ignore terms that occur in a larger share of documents
    Rows are L2-normalized (scikit-learn default), so the dot product of two
    rows is their cosine similarity.
    """
    vectorizer = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,
        lowercase=False,  # already lowercase after strong cleaning
        ngram_range=tuple(ngram_range),
        min_df=min_df,
        max_df=max_df,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix.tocsr()
