"""Extract topics with both methods: NMF on TF-IDF and KMeans on embeddings.

Run from the project root after the vectorization pipeline:
    python -m src.topics.pipeline            # k from the config (k_initial)
    python -m src.topics.pipeline --k 12

Writes one table per method to results/tables/ (topic words, share of
reports, share of reports with each state marker) and the topic and state
markers of every report to results/cache/topics.parquet.
"""

import argparse
import logging

import pandas as pd

from src.config import load_config, project_path
from src.topics.kmeans import fit_kmeans
from src.topics.nmf import assign_topics, fit_nmf
from src.topics.states import state_markers, topic_states
from src.topics.words import class_tfidf, top_terms, topic_table
from src.vectorization.pipeline import embeddings_from_config, load_processed, tfidf_from_config

logger = logging.getLogger(__name__)


def extract_topics(reports: pd.DataFrame, cfg: dict, k: int) -> dict:
    """Fit both methods with k topics and return labels, topic words and tables.

    The tables contain one row per topic with its words, its share of the
    reports and the share of its reports with each state marker.
    """
    topics_cfg = cfg["topics"]
    n_words = topics_cfg["n_top_words"]

    vectorizer, tfidf = tfidf_from_config(reports, cfg)
    terms = vectorizer.get_feature_names_out()

    nmf_model, nmf_weights = fit_nmf(
        tfidf, k, cfg["seed"], topics_cfg["nmf"]["init"], topics_cfg["nmf"]["max_iter"]
    )
    nmf_labels = assign_topics(nmf_weights)
    nmf_words = top_terms(nmf_model.components_, terms, n_words)
    logger.info("NMF: %d iterations, %d reports without topic", nmf_model.n_iter_, (nmf_labels < 0).sum())

    embeddings = embeddings_from_config(reports, cfg)
    kmeans_model, kmeans_labels = fit_kmeans(embeddings, k, cfg["seed"], topics_cfg["kmeans"]["n_init"])
    kmeans_words = top_terms(class_tfidf(reports["text_strong"], kmeans_labels, k, vectorizer), terms, n_words)
    logger.info("KMeans: %d iterations", kmeans_model.n_iter_)

    markers = state_markers(reports["text_strong"], cfg["states"])
    logger.info("State markers (share of all reports): %s", markers.mean().round(3).to_dict())

    return {
        "vectorizer": vectorizer,
        "tfidf": tfidf,
        "embeddings": embeddings,
        "nmf_model": nmf_model,
        "nmf_weights": nmf_weights,
        "nmf_labels": nmf_labels,
        "nmf_table": with_states(topic_table(nmf_labels, nmf_words), nmf_labels, markers),
        "kmeans_model": kmeans_model,
        "kmeans_labels": kmeans_labels,
        "kmeans_table": with_states(topic_table(kmeans_labels, kmeans_words), kmeans_labels, markers),
        "markers": markers,
    }


def with_states(table: pd.DataFrame, labels, markers: pd.DataFrame) -> pd.DataFrame:
    """Add the share of reports with each state marker to a topic table."""
    return table.merge(topic_states(labels, markers), on="topic", how="left")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, help="number of topics (default: topics.k_initial)")
    args = parser.parse_args()

    cfg = load_config()
    k = args.k or cfg["topics"]["k_initial"]
    reports = load_processed(cfg)
    result = extract_topics(reports, cfg, k)

    tables_dir = project_path(cfg["results"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    for method in ("nmf", "kmeans"):
        path = tables_dir / f"{method}_topics_k{k}.csv"
        result[f"{method}_table"].to_csv(path, index=False, encoding="utf-8")
        logger.info("Saved %s", path)

    assignments = pd.DataFrame(
        {
            "service_request_id": reports["service_request_id"],
            "nmf_topic": result["nmf_labels"],
            "kmeans_topic": result["kmeans_labels"],
        }
    ).join(result["markers"])
    path = project_path(cfg["results"]["assignments_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(path, index=False)
    logger.info("Saved topic assignments (k=%d) to %s", k, path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raise SystemExit(main())
