"""Topic extraction method 1: NMF on the TF-IDF matrix.

NMF approximates the TF-IDF matrix X (reports x terms) as the product of two
non-negative matrices: X ~ W @ H.
- W (reports x topics): how strongly each report belongs to each topic
- H (topics x terms): how strongly each term belongs to each topic
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

# Label for reports that cannot be assigned to a topic.
NO_TOPIC = -1


def fit_nmf(tfidf: csr_matrix, k: int, seed: int, init: str, max_iter: int) -> tuple[NMF, np.ndarray]:
    """Fit NMF with k topics and return the model and W."""
    model = NMF(n_components=k, init=init, max_iter=max_iter, random_state=seed)
    weights = model.fit_transform(tfidf)
    return model, weights


def assign_topics(weights: np.ndarray) -> np.ndarray:
    """Assign each report to the topic with the highest weight.

    Reports whose weights are all zero (no TF-IDF term) get NO_TOPIC. Without
    this check, argmax would silently assign them to topic 0.
    """
    labels = weights.argmax(axis=1)
    labels[weights.max(axis=1) == 0] = NO_TOPIC
    return labels


def top_reports(weights: np.ndarray, labels: np.ndarray, n: int) -> dict[int, np.ndarray]:
    """Indices of the n reports with the highest weight for each topic (typical examples).

    Only reports assigned to the topic are considered.
    """
    result = {}
    for topic in range(weights.shape[1]):
        members = np.flatnonzero(labels == topic)
        order = np.argsort(-weights[members, topic], kind="stable")[:n]
        result[topic] = members[order]
    return result
