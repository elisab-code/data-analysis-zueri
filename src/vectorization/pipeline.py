"""Vectorize the processed reports: TF-IDF (strong texts) and embeddings (light texts).

Run from the project root after the preprocessing pipeline:
    python -m src.vectorization.pipeline

TF-IDF takes a few seconds and is rebuilt where it is needed. The embeddings
are cached (see embeddings.py), so later steps load them without re-encoding.
"""

import logging

import pandas as pd

from src.config import load_config, project_path
from src.vectorization.compare import describe
from src.vectorization.embeddings import load_or_encode
from src.vectorization.tfidf import build_tfidf

logger = logging.getLogger(__name__)


def load_processed(cfg: dict) -> pd.DataFrame:
    """Read the output of the preprocessing pipeline."""
    return pd.read_parquet(project_path(cfg["data"]["processed_file"]))


def tfidf_from_config(reports: pd.DataFrame, cfg: dict):
    """Build the TF-IDF matrix with the parameters from the config."""
    tfidf_cfg = cfg["tfidf"]
    return build_tfidf(
        reports["text_strong"], tfidf_cfg["ngram_range"], tfidf_cfg["min_df"], tfidf_cfg["max_df"]
    )


def embeddings_from_config(reports: pd.DataFrame, cfg: dict):
    """Load the cached embeddings or encode the light texts."""
    emb_cfg = cfg["embeddings"]
    return load_or_encode(
        reports["text_light"].tolist(),
        emb_cfg["model"],
        emb_cfg["revision"],
        emb_cfg["batch_size"],
        project_path(emb_cfg["cache_file"]),
    )


def main() -> int:
    cfg = load_config()
    reports = load_processed(cfg)

    _, tfidf = tfidf_from_config(reports, cfg)
    logger.info("TF-IDF: %s", describe(tfidf))

    embeddings = embeddings_from_config(reports, cfg)
    logger.info("Embeddings: %s", describe(embeddings))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # The Hugging Face client logs every HTTP request on INFO level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raise SystemExit(main())
