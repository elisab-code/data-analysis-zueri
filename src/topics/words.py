"""Topic words and topic shares for both methods.

Both methods describe a topic by a ranked list of terms from the same
TF-IDF vocabulary, so their topics can be compared directly.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.topics.nmf import NO_TOPIC


def top_terms(topic_term: np.ndarray, terms: Sequence[str], n: int) -> list[list[str]]:
    """The n terms with the highest weight for each topic (one row per topic)."""
    order = np.argsort(-topic_term, axis=1, kind="stable")[:, :n]
    return [[terms[i] for i in row] for row in order]


def class_tfidf(
    texts: Sequence[str], labels: np.ndarray, n_clusters: int, vectorizer: TfidfVectorizer
) -> np.ndarray:
    """Class-based TF-IDF (c-TF-IDF, as in BERTopic): one row of term weights per cluster.

    All texts of a cluster are treated as one document. A term gets a high weight
    if it is frequent in the cluster but rare in the other clusters:
        weight(t, c) = tf(t, c) * log(1 + A / f(t))
    tf(t, c): frequency of term t in cluster c, divided by the number of terms in c
    A: average number of terms per cluster, f(t): frequency of t in all clusters.
    The vocabulary is taken from the fitted TF-IDF vectorizer, so both methods use
    the same terms. Reports with NO_TOPIC are ignored.
    """
    counter = CountVectorizer(
        vocabulary=vectorizer.vocabulary_,
        token_pattern=vectorizer.token_pattern,
        ngram_range=vectorizer.ngram_range,
        lowercase=False,
    )
    counts = counter.transform(texts)

    # Membership matrix (clusters x reports): 1 if the report belongs to the cluster.
    # Multiplied with the counts (reports x terms), it sums the counts per cluster.
    valid = np.flatnonzero(labels != NO_TOPIC)
    membership = csr_matrix(
        (np.ones(len(valid)), (labels[valid], valid)), shape=(n_clusters, len(labels))
    )
    cluster_counts = (membership @ counts).toarray().astype(float)

    terms_per_cluster = cluster_counts.sum(axis=1, keepdims=True)
    tf = np.divide(cluster_counts, terms_per_cluster, out=np.zeros_like(cluster_counts), where=terms_per_cluster > 0)

    term_totals = cluster_counts.sum(axis=0)
    average = terms_per_cluster.mean()
    ratio = np.divide(average, term_totals, out=np.zeros_like(term_totals), where=term_totals > 0)
    return tf * np.log1p(ratio)


def topic_table(labels: np.ndarray, words: list[list[str]]) -> pd.DataFrame:
    """One row per topic with number and share of reports and the topic words.

    Sorted by size. Reports without topic (NO_TOPIC) are listed as the last row.
    """
    counts = pd.Series(labels).value_counts()
    table = pd.DataFrame({"topic": counts.index, "reports": counts.to_numpy()})
    table["share"] = table["reports"] / len(labels)
    table["words"] = [
        "(no topic)" if topic == NO_TOPIC else ", ".join(words[topic]) for topic in table["topic"]
    ]
    table["is_topic"] = table["topic"] != NO_TOPIC
    table = table.sort_values(["is_topic", "reports", "topic"], ascending=[False, False, True])
    return table.drop(columns="is_topic").reset_index(drop=True)
