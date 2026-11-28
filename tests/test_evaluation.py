"""Tests for src/evaluation (quality measures and the k scan figure)."""

import numpy as np
import pandas as pd
import pytest
from gensim.corpora import Dictionary
from sklearn.decomposition import NMF

from src.evaluation.k_scan import k_values, plot_scan
from src.evaluation.metrics import (
    coherence_cv,
    largest_share,
    stability,
    topic_diversity,
    unassigned_share,
    unigrams,
)
from src.topics.nmf import NO_TOPIC, assign_topics, topic_contributions


def test_unigrams_splits_pairs_and_removes_repeats():
    assert unigrams(["lampe defekt", "lampe", "dunkel", "licht"], 3) == ["lampe", "defekt", "dunkel"]


def test_topic_diversity():
    assert topic_diversity([["a", "b"], ["c", "d"]]) == 1.0
    assert topic_diversity([["a", "b"], ["a", "c"]]) == 0.75


def test_stability_ignores_topic_numbers():
    a = np.array([0, 0, 1, 1, 2, 2])
    renamed = np.array([2, 2, 0, 0, 1, 1])
    assert stability([a, renamed, a]) == pytest.approx(1.0)


def test_stability_is_low_for_unrelated_assignments():
    rng = np.random.default_rng(0)
    runs = [rng.integers(0, 5, 2000) for _ in range(3)]
    assert abs(stability(runs)) < 0.05


def test_largest_and_unassigned_share():
    labels = np.array([0, 0, 0, 1, NO_TOPIC])
    assert largest_share(labels) == pytest.approx(0.6)
    assert unassigned_share(labels) == pytest.approx(0.2)
    assert largest_share(np.array([NO_TOPIC, NO_TOPIC])) == 0.0


def test_coherence_prefers_words_that_occur_together():
    texts = [["abfall", "sack"], ["abfall", "sack"], ["lampe", "licht"], ["lampe", "licht"]] * 5
    dictionary = Dictionary(texts)
    good = coherence_cv([["abfall", "sack"], ["lampe", "licht"]], texts, dictionary)
    bad = coherence_cv([["abfall", "licht"], ["lampe", "sack"]], texts, dictionary)
    assert good > bad


def test_assign_topics_with_min_weight():
    weights = np.array([[0.5, 0.1], [0.05, 0.02], [0.0, 0.0]])
    assert assign_topics(weights, min_weight=0.1).tolist() == [0, NO_TOPIC, NO_TOPIC]
    assert assign_topics(weights).tolist() == [0, 0, NO_TOPIC]


def test_topic_contributions_scale_by_topic_length():
    model = NMF(n_components=2)
    model.components_ = np.array([[3.0, 4.0], [1.0, 0.0]])  # lengths 5 and 1
    weights = np.array([[0.1, 0.4]])
    np.testing.assert_allclose(topic_contributions(model, weights), [[0.5, 0.4]])


def test_k_values():
    assert k_values({"k_min": 5, "k_max": 20, "k_step": 5}) == [5, 10, 15, 20]


def test_plot_scan_writes_file(tmp_path):
    table = pd.DataFrame(
        [
            {"method": m, "k": k, "coherence": 0.5, "diversity": 0.8, "stability": 0.6,
             "largest_share": 0.2, "unassigned_share": 0.1}
            for m in ("NMF", "KMeans") for k in (5, 10)
        ]
    )
    path = tmp_path / "k_scan.png"
    plot_scan(table, path)
    assert path.stat().st_size > 0
