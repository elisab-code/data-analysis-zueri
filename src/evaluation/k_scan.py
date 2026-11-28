"""k scan: quality of both topic methods for several numbers of topics.

Run from the project root after the vectorization pipeline (about 20 minutes,
mostly KMeans for large k):
    python -m src.evaluation.k_scan
    python -m src.evaluation.k_scan --k 10 20     # only these k

For every k and method it measures:
- coherence (c_v): do the top words of a topic occur together in the texts?
- diversity: are the topics different from each other?
- stability: do fits on different samples of the reports group them the same way?
  For NMF without the min_weight threshold, otherwise the many reports without
  a clear topic would agree in every run and make NMF look more stable.
- largest_share: share of all reports in the largest topic (catch-all topics)
- unassigned_share: share of reports without a clear topic (NMF only)

Writes results/tables/k_scan.csv and results/figures/k_scan.png.
"""

import argparse
import logging
import time

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary

from src.config import load_config, project_path
from src.evaluation.metrics import (
    coherence_cv,
    largest_share,
    stability,
    topic_diversity,
    unassigned_share,
    unigrams,
)
from src.topics.kmeans import fit_kmeans
from src.topics.nmf import assign_topics, fit_nmf, topic_contributions
from src.topics.words import class_tfidf, top_terms
from src.vectorization.pipeline import embeddings_from_config, load_processed, tfidf_from_config

logger = logging.getLogger(__name__)

METHODS = ["NMF", "KMeans"]
# First two colors of the validated categorical palette (blue, orange).
COLORS = {"NMF": "#2a78d6", "KMeans": "#eb6834"}
MARKERS = {"NMF": "o", "KMeans": "s"}
PANELS = {
    "coherence": "Coherence (c_v)",
    "diversity": "Topic diversity",
    "stability": "Stability (mean ARI)",
    "largest_share": "Largest topic (share of reports)",
    "unassigned_share": "No clear topic (share of reports)",
}


def k_values(topics_cfg: dict) -> list[int]:
    """k_min, k_min + k_step, ..., k_max from the config."""
    return list(range(topics_cfg["k_min"], topics_cfg["k_max"] + 1, topics_cfg["k_step"]))


def nmf_labels(tfidf, k: int, seed: int, nmf_cfg: dict, fit_rows=None, min_weight=None):
    """Fit NMF (on all reports or only on fit_rows) and assign all reports.

    Returns the model and the topic of every report. min_weight: threshold for
    a clear topic, by default the one from the config.
    """
    data = tfidf if fit_rows is None else tfidf[fit_rows]
    model, weights = fit_nmf(data, k, seed, nmf_cfg["init"], nmf_cfg["max_iter"])
    if fit_rows is not None:
        weights = model.transform(tfidf)  # weights for all reports
    if min_weight is None:
        min_weight = nmf_cfg["min_weight"]
    return model, assign_topics(topic_contributions(model, weights), min_weight)


def kmeans_labels(embeddings: np.ndarray, k: int, seed: int, n_init: int, fit_rows=None):
    """Fit KMeans (on all reports or only on fit_rows) and assign all reports."""
    if fit_rows is None:
        return fit_kmeans(embeddings, k, seed, n_init)
    model, _ = fit_kmeans(embeddings[fit_rows], k, seed, n_init)
    return model, model.predict(embeddings)


def scan(reports: pd.DataFrame, cfg: dict, ks: list[int]) -> pd.DataFrame:
    """One row per k and method with all quality measures."""
    topics_cfg = cfg["topics"]
    nmf_cfg = topics_cfg["nmf"]
    n_init = topics_cfg["kmeans"]["n_init"]
    n_words = topics_cfg["n_top_words"]

    vectorizer, tfidf = tfidf_from_config(reports, cfg)
    terms = vectorizer.get_feature_names_out()
    embeddings = embeddings_from_config(reports, cfg)
    texts = [text.split() for text in reports["text_strong"]]
    dictionary = Dictionary(texts)

    # One random sample of reports per seed, the same samples for every k and method.
    n_sample = int(topics_cfg["stability_sample"] * len(reports))
    samples = [
        np.sort(np.random.default_rng(seed).choice(len(reports), n_sample, replace=False))
        for seed in topics_cfg["stability_seeds"]
    ]

    rows = []
    for k in ks:
        start = time.perf_counter()

        model, labels = nmf_labels(tfidf, k, cfg["seed"], nmf_cfg)
        nmf_words = top_terms(model.components_, terms, n_words)
        nmf_runs = [
            nmf_labels(tfidf, k, seed, nmf_cfg, sample, min_weight=0.0)[1]
            for seed, sample in zip(topics_cfg["stability_seeds"], samples)
        ]
        rows.append(measure("NMF", k, labels, nmf_words, nmf_runs, texts, dictionary, n_words))

        _, labels = kmeans_labels(embeddings, k, cfg["seed"], n_init)
        kmeans_words = top_terms(class_tfidf(reports["text_strong"], labels, k, vectorizer), terms, n_words)
        kmeans_runs = [
            kmeans_labels(embeddings, k, seed, n_init, sample)[1]
            for seed, sample in zip(topics_cfg["stability_seeds"], samples)
        ]
        rows.append(measure("KMeans", k, labels, kmeans_words, kmeans_runs, texts, dictionary, n_words))

        logger.info("k=%d done in %.0f s: %s", k, time.perf_counter() - start, rows[-2:])
    return pd.DataFrame(rows)


def measure(method, k, labels, words, runs, texts, dictionary, n_words) -> dict:
    """All quality measures for one fit (labels, words) and its sample runs."""
    topics = [unigrams(topic, n_words) for topic in words]
    return {
        "method": method,
        "k": k,
        "coherence": coherence_cv(topics, texts, dictionary),
        "diversity": topic_diversity(topics),
        "stability": stability(runs),
        "largest_share": largest_share(labels),
        "unassigned_share": unassigned_share(labels),
    }


def plot_scan(table: pd.DataFrame, path) -> None:
    """One small chart per measure, k on the x axis, one line per method."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(18, 3.6), sharex=True)
    for ax, (column, title) in zip(axes, PANELS.items()):
        for method in METHODS:
            part = table[table["method"] == method]
            ax.plot(
                part["k"], part[column], color=COLORS[method], marker=MARKERS[method],
                markersize=6, linewidth=2, label=method,
            )
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("number of topics k")
        ax.set_xticks(sorted(table["k"].unique()))
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, nargs="+", help="k values (default: from the config)")
    args = parser.parse_args()

    cfg = load_config()
    ks = args.k or k_values(cfg["topics"])
    logger.info("k scan for k = %s", ks)
    table = scan(load_processed(cfg), cfg, ks)

    tables_dir = project_path(cfg["results"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "k_scan.csv", index=False, encoding="utf-8")

    figures_dir = project_path(cfg["results"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_scan(table, figures_dir / "k_scan.png")
    logger.info("Saved k_scan.csv and k_scan.png")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("gensim").setLevel(logging.WARNING)  # logs every coherence step
    raise SystemExit(main())
