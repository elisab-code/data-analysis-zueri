"""Key figures for comparing the two vector representations.

Both representations are L2-normalized row by row, so the dot product of two
rows is their cosine similarity. This works for sparse (TF-IDF) and dense
(embedding) matrices alike.
"""

import numpy as np
from scipy.sparse import issparse


def describe(matrix) -> dict:
    """Shape, share of non-zero values, memory and number of all-zero rows."""
    n_rows, n_dims = matrix.shape
    if issparse(matrix):
        non_zero = matrix.nnz
        memory = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        zero_rows = int((matrix.getnnz(axis=1) == 0).sum())
    else:
        non_zero = int(np.count_nonzero(matrix))
        memory = matrix.nbytes
        zero_rows = int((~matrix.any(axis=1)).sum())
    return {
        "rows": n_rows,
        "dimensions": n_dims,
        "density": non_zero / (n_rows * n_dims),
        "memory_mb": memory / 1e6,
        "zero_rows": zero_rows,
    }


def nearest_neighbors(matrix, query: int, k: int) -> np.ndarray:
    """Indices of the k rows most similar to row `query` (the query itself excluded).

    Ties are broken by index, so the result is deterministic.
    """
    scores = matrix @ matrix[query].T
    scores = scores.toarray().ravel() if issparse(scores) else np.asarray(scores).ravel()
    scores[query] = -np.inf
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:k]


def neighbor_overlap(matrix_a, matrix_b, queries, k: int) -> float:
    """Mean share of shared neighbors among the k nearest neighbors (0 = none, 1 = all)."""
    shared = [
        len(set(nearest_neighbors(matrix_a, q, k)) & set(nearest_neighbors(matrix_b, q, k))) / k
        for q in queries
    ]
    return float(np.mean(shared))
