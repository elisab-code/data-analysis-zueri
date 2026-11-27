"""Tests for src/vectorization (TF-IDF, embedding cache, comparison figures).

The embedding tests use a fake encoder, so no model is downloaded.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.vectorization.compare import describe, nearest_neighbors, neighbor_overlap
from src.vectorization.embeddings import fingerprint, load_or_encode
from src.vectorization.tfidf import build_tfidf, is_excluded

TEXTS = [
    "e-scooter trottoir",
    "e-scooter trottoir nicht",
    "tempo 30 nicht",
    "tempo 30 abfall",
]


# --- TF-IDF --------------------------------------------------------------


def test_tfidf_keeps_hyphen_compounds_and_numbers():
    vectorizer, _ = build_tfidf(TEXTS, (1, 1), 1, 1.0)
    vocabulary = set(vectorizer.get_feature_names_out())
    assert {"e-scooter", "30"} <= vocabulary


def test_tfidf_builds_word_pairs():
    vectorizer, _ = build_tfidf(TEXTS, (1, 2), 1, 1.0)
    assert "tempo 30" in vectorizer.get_feature_names_out()


def test_tfidf_min_df_removes_rare_terms():
    vectorizer, _ = build_tfidf(TEXTS, (1, 1), 2, 1.0)
    assert "abfall" not in vectorizer.get_feature_names_out()  # occurs in 1 text only


def test_tfidf_shape_and_normalized_rows():
    vectorizer, matrix = build_tfidf(TEXTS, (1, 2), 1, 1.0)
    assert matrix.shape == (len(TEXTS), len(vectorizer.get_feature_names_out()))
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    np.testing.assert_allclose(norms, 1.0)


def test_tfidf_empty_text_gives_zero_row():
    _, matrix = build_tfidf([*TEXTS, ""], (1, 1), 1, 1.0)
    assert matrix[len(TEXTS)].nnz == 0


@pytest.mark.parametrize(
    ("term", "excluded"),
    [
        ("seit", True),
        ("stehen seit", True),  # a pair with an excluded word
        ("2", True),  # a number alone
        ("10 12", True),
        ("10.07.2024", True),
        ("stehen", False),
        ("tempo 30", False),  # a number with a word
        ("e-scooter", False),
    ],
)
def test_is_excluded(term, excluded):
    assert is_excluded(term, frozenset({"seit", "nicht"}), exclude_numbers=True) == excluded


def test_is_excluded_keeps_numbers_if_not_requested():
    assert not is_excluded("30", frozenset(), exclude_numbers=False)


def test_tfidf_exclusions_remove_terms():
    vectorizer, _ = build_tfidf(TEXTS, (1, 2), 1, 1.0, exclude_words=["nicht"], exclude_numbers=True)
    terms = set(vectorizer.get_feature_names_out())
    assert not {"nicht", "trottoir nicht", "30", "30 nicht"} & terms
    assert {"trottoir", "tempo 30", "e-scooter trottoir"} <= terms


def test_tfidf_exclusions_keep_idf_and_normalize_rows():
    full, full_matrix = build_tfidf(TEXTS, (1, 1), 1, 1.0)
    reduced, matrix = build_tfidf(TEXTS, (1, 1), 1, 1.0, exclude_words=["nicht"])
    # The IDF of a term depends only on how many texts contain it.
    full_idf = dict(zip(full.get_feature_names_out(), full.idf_))
    for term, idf in zip(reduced.get_feature_names_out(), reduced.idf_):
        assert idf == pytest.approx(full_idf[term])
    # Rows are normalized over the remaining terms.
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    np.testing.assert_allclose(norms, 1.0)
    assert matrix.shape[1] == full_matrix.shape[1] - 1


# --- embedding cache -------------------------------------------------------


class FakeEncoder:
    """Stands in for the model: returns fixed vectors and counts the calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts, model_name, revision, batch_size):
        self.calls += 1
        return np.ones((len(texts), 3), dtype=np.float32)


def test_fingerprint_changes_with_texts_model_and_revision():
    base = fingerprint(["a", "b"], "model", "rev1")
    assert base == fingerprint(["a", "b"], "model", "rev1")
    assert base != fingerprint(["b", "a"], "model", "rev1")
    assert base != fingerprint(["a", "b"], "other", "rev1")
    assert base != fingerprint(["a", "b"], "model", "rev2")


def test_fingerprint_does_not_merge_neighbouring_texts():
    assert fingerprint(["ab", "c"], "model", None) != fingerprint(["a", "bc"], "model", None)


def test_cache_is_used_on_second_call(tmp_path):
    encoder = FakeEncoder()
    cache = tmp_path / "emb.npy"
    first = load_or_encode(["a", "b"], "model", None, 2, cache, encoder)
    second = load_or_encode(["a", "b"], "model", None, 2, cache, encoder)
    assert encoder.calls == 1
    np.testing.assert_array_equal(first, second)


def test_cache_is_renewed_when_texts_change(tmp_path):
    encoder = FakeEncoder()
    cache = tmp_path / "emb.npy"
    load_or_encode(["a", "b"], "model", None, 2, cache, encoder)
    result = load_or_encode(["a", "b", "c"], "model", None, 2, cache, encoder)
    assert encoder.calls == 2
    assert result.shape == (3, 3)


# --- comparison --------------------------------------------------------------

# Three normalized vectors: rows 0 and 1 are similar, row 2 points elsewhere.
VECTORS = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])


def test_describe_dense_and_sparse_agree():
    dense, sparse = describe(VECTORS), describe(csr_matrix(VECTORS))
    for key in ("rows", "dimensions", "density", "zero_rows"):
        assert dense[key] == pytest.approx(sparse[key])
    assert dense["density"] == pytest.approx(4 / 6)


def test_describe_counts_zero_rows():
    assert describe(csr_matrix([[0.0, 0.0], [1.0, 0.0]]))["zero_rows"] == 1


def test_nearest_neighbors_excludes_query():
    assert nearest_neighbors(VECTORS, 0, 2).tolist() == [1, 2]
    assert nearest_neighbors(csr_matrix(VECTORS), 2, 1).tolist() == [1]


def test_neighbor_overlap_of_identical_matrices_is_one():
    assert neighbor_overlap(VECTORS, csr_matrix(VECTORS), [0, 1, 2], 1) == 1.0
