"""Topic extraction method 2: KMeans on the sentence embeddings.

The embeddings are normalized to length 1. For such vectors, a smaller
Euclidean distance means a higher similarity of meaning, so KMeans groups
reports by meaning. The clusters have no words of their own. Their topic
words are computed from the texts in each cluster (see words.py).
"""

import numpy as np
from sklearn.cluster import KMeans


def fit_kmeans(embeddings: np.ndarray, k: int, seed: int, n_init: int) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans with k clusters and return the model and the cluster of each report.

    KMeans (Lloyd 1982) with the scikit-learn default start "k-means++"
    (Arthur and Vassilvitskii 2007), which spreads the start centers apart.
    n_init: number of runs with different start centers. The best run is kept.
    """
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = model.fit_predict(embeddings)
    return model, labels


def central_reports(embeddings: np.ndarray, labels: np.ndarray, centers: np.ndarray, n: int) -> dict[int, np.ndarray]:
    """Indices of the n reports closest to the center of each cluster (typical examples)."""
    result = {}
    for cluster, center in enumerate(centers):
        members = np.flatnonzero(labels == cluster)
        distances = np.linalg.norm(embeddings[members] - center, axis=1)
        result[cluster] = members[np.argsort(distances, kind="stable")[:n]]
    return result
