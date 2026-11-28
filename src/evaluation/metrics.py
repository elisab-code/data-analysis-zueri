"""Quality measures for topics: coherence, diversity, stability and topic sizes."""

from collections.abc import Sequence
from itertools import combinations

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import adjusted_rand_score

from src.topics.nmf import NO_TOPIC


def unigrams(terms: Sequence[str], n: int) -> list[str]:
    """Split word pairs into single words and keep the first n distinct words.

    Example: ["lampe defekt", "lampe", "dunkel"] -> ["lampe", "defekt", "dunkel"].
    Coherence and diversity count single words, so a pair and its parts are
    not counted twice.
    """
    words = dict.fromkeys(word for term in terms for word in term.split(" "))
    return list(words)[:n]


def coherence_cv(topics: list[list[str]], texts: list[list[str]], dictionary: Dictionary) -> float:
    """Mean c_v coherence of the topics (higher = top words occur together more often).

    c_v (Röder et al. 2015) checks how often the top words of a topic occur
    close to each other in the texts: co-occurrence in a sliding window of
    110 words, compared via NPMI vectors and the cosine, averaged over the
    words. Values lie between 0 and 1.
    processes=1: gensim's parallel mode is unreliable on Windows.
    """
    model = CoherenceModel(
        topics=topics, texts=texts, dictionary=dictionary, coherence="c_v", processes=1
    )
    return model.get_coherence()


def topic_diversity(topics: list[list[str]]) -> float:
    """Share of distinct words among the top words of all topics (1 = no word repeated).

    Following Dieng et al. (2020), who use the top 25 words per topic. Here the
    top n_top_words from the config (10) are used, the same words as for the
    coherence. Low values mean that topics share many words.
    """
    words = [word for topic in topics for word in topic]
    return len(set(words)) / len(words)


def stability(label_sets: list[np.ndarray]) -> float:
    """Mean adjusted Rand index (ARI) over all pairs of topic assignments.

    The assignments come from fits on different samples of the reports.
    ARI 1: the same reports are grouped together in both, 0: agreement by chance.
    The topic numbers themselves may differ between fits, ARI ignores them.

    Based on Greene et al. (2014), who also fit models on samples of the data to
    choose k. They compare the top terms of the topics (average Jaccard, topics
    matched with the Hungarian method). Here the topic assignments of the
    reports are compared instead (ARI, Hubert and Arabie 1985), so NMF and
    KMeans are measured on the same scale.
    """
    scores = [adjusted_rand_score(a, b) for a, b in combinations(label_sets, 2)]
    return float(np.mean(scores))


def largest_share(labels: np.ndarray) -> float:
    """Share of all reports in the largest topic (reports without topic excluded)."""
    assigned = labels[labels != NO_TOPIC]
    if len(assigned) == 0:
        return 0.0
    return np.bincount(assigned).max() / len(labels)


def unassigned_share(labels: np.ndarray) -> float:
    """Share of reports without a clear topic (NO_TOPIC)."""
    return float(np.mean(labels == NO_TOPIC))
