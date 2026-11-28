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
    """Fit NMF with k topics and return the model and W.

    NMF as introduced by Lee and Seung (1999). scikit-learn fits it with
    coordinate descent (its default solver), not with the multiplicative updates
    of Lee and Seung. init "nndsvda": start from an SVD (Boutsidis and
    Gallopoulos 2008), with zeros replaced by the mean of the data.
    """
    model = NMF(n_components=k, init=init, max_iter=max_iter, random_state=seed)
    weights = model.fit_transform(tfidf)
    return model, weights


def topic_contributions(model: NMF, weights: np.ndarray) -> np.ndarray:
    """W scaled by the length of each topic's term vector (row of H).

    The rows of H have different lengths, so raw weights of different topics
    are not comparable. After scaling, a value is the length of the part of
    the report's TF-IDF vector that the topic describes. A report vector has
    length 1, so 0.3 means the topic describes a large part of the report.
    """
    return weights * np.linalg.norm(model.components_, axis=1)


def assign_topics(weights: np.ndarray, min_weight: float = 0.0) -> np.ndarray:
    """Assign each report to the topic with the highest weight.

    Reports whose highest weight is zero (no TF-IDF term) or below min_weight
    get NO_TOPIC: no topic describes them clearly. Without this check, argmax
    would silently assign reports without any term to topic 0.
    """
    labels = weights.argmax(axis=1)
    best = weights.max(axis=1)
    labels[(best == 0) | (best < min_weight)] = NO_TOPIC
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
