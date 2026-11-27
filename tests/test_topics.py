"""Tests for src/topics (NMF, KMeans, topic words and tables) on small examples."""

import numpy as np
import pytest

from src.topics.kmeans import central_reports, fit_kmeans
from src.topics.nmf import NO_TOPIC, assign_topics, fit_nmf, top_reports
from src.topics.words import class_tfidf, top_terms, topic_table
from src.vectorization.tfidf import build_tfidf

# Two clear topics (lamps and waste) and one report without any term.
TEXTS = [
    "lampe defekt",
    "lampe dunkel defekt",
    "strassenlampe lampe dunkel",
    "abfall liegen",
    "abfall sack liegen",
    "abfall sack trottoir",
    "",
]


@pytest.fixture
def tfidf():
    return build_tfidf(TEXTS, (1, 1), 1, 1.0)


# --- NMF -----------------------------------------------------------------


def test_assign_topics_marks_zero_rows():
    weights = np.array([[0.2, 0.5], [0.0, 0.0], [0.3, 0.1]])
    assert assign_topics(weights).tolist() == [1, NO_TOPIC, 0]


def test_nmf_separates_two_topics(tfidf):
    _, matrix = tfidf
    _, weights = fit_nmf(matrix, 2, seed=0, init="nndsvda", max_iter=500)
    labels = assign_topics(weights)
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:6])) == 1
    assert labels[0] != labels[3]
    assert labels[6] == NO_TOPIC


def test_nmf_is_deterministic(tfidf):
    _, matrix = tfidf
    first = fit_nmf(matrix, 2, seed=0, init="nndsvda", max_iter=500)[1]
    second = fit_nmf(matrix, 2, seed=0, init="nndsvda", max_iter=500)[1]
    np.testing.assert_array_equal(first, second)


def test_top_reports_only_from_own_topic():
    weights = np.array([[0.9, 0.0], [0.5, 0.1], [0.1, 0.8], [0.0, 0.0]])
    result = top_reports(weights, assign_topics(weights), 5)
    assert result[0].tolist() == [0, 1]
    assert result[1].tolist() == [2]


# --- KMeans --------------------------------------------------------------

# Normalized 2D points around two directions.
POINTS = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]])


def test_kmeans_groups_similar_points():
    _, labels = fit_kmeans(POINTS, 2, seed=0, n_init=5)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_kmeans_is_deterministic():
    first = fit_kmeans(POINTS, 2, seed=0, n_init=5)[1]
    second = fit_kmeans(POINTS, 2, seed=0, n_init=5)[1]
    np.testing.assert_array_equal(first, second)


def test_central_reports_returns_closest_first():
    labels = np.array([0, 0, 0])
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.5, 0.0]])
    result = central_reports(points, labels, np.array([[0.0, 0.0]]), 2)
    assert result[0].tolist() == [0, 2]


# --- topic words and tables ------------------------------------------------


def test_top_terms_ranks_by_weight():
    weights = np.array([[0.1, 0.7, 0.2], [0.5, 0.0, 0.4]])
    assert top_terms(weights, ["a", "b", "c"], 2) == [["b", "c"], ["a", "c"]]


def test_class_tfidf_prefers_terms_typical_for_a_cluster(tfidf):
    vectorizer, _ = tfidf
    labels = np.array([0, 0, 0, 1, 1, 1, NO_TOPIC])
    weights = class_tfidf(TEXTS, labels, 2, vectorizer)
    words = top_terms(weights, vectorizer.get_feature_names_out(), 2)
    assert weights.shape == (2, len(vectorizer.get_feature_names_out()))
    assert set(words[0]) <= {"lampe", "dunkel", "defekt", "strassenlampe"}
    assert set(words[1]) <= {"abfall", "sack", "liegen", "trottoir"}


def test_class_tfidf_keeps_empty_cluster_as_zero_row(tfidf):
    vectorizer, _ = tfidf
    labels = np.array([0, 0, 0, 0, 0, 0, 0])
    weights = class_tfidf(TEXTS, labels, 2, vectorizer)
    assert not weights[1].any()


def test_topic_table_sorted_with_no_topic_row():
    labels = np.array([1, 1, 0, NO_TOPIC, 1])
    table = topic_table(labels, [["abfall"], ["lampe", "defekt"]])
    assert table["topic"].tolist() == [1, 0, NO_TOPIC]
    assert table["reports"].tolist() == [3, 1, 1]
    assert table.loc[0, "words"] == "lampe, defekt"
    assert table.loc[2, "words"] == "(no topic)"
    assert table["share"].sum() == pytest.approx(1.0)
