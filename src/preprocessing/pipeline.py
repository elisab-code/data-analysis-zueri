"""Preprocessing pipeline: raw CSV -> cleaned texts (light and strong) as Parquet.

Run from the project root:
    python -m src.preprocessing.pipeline
"""

import logging

import pandas as pd
from spacy.language import Language

from src.config import load_config, project_path
from src.preprocessing.clean import fix_encoding, light_clean
from src.preprocessing.strong import load_nlp, strong_clean

logger = logging.getLogger(__name__)

# Columns taken from the raw data. The text column is added from the config.
META_COLUMNS = ["service_request_id", "requested_datetime", "service_code", "interface_used"]


def load_raw(path, text_column: str) -> pd.DataFrame:
    """Read the raw CSV (UTF-8 with BOM) with the metadata and the text column."""
    return pd.read_csv(path, encoding="utf-8-sig", usecols=[*META_COLUMNS, text_column])


def prepare(df: pd.DataFrame, text_column: str, pre_cfg: dict, nlp: Language) -> pd.DataFrame:
    """Clean the reports and return one row per report with text_light and text_strong.

    Steps:
    1. drop reports without text
    2. repair the encoding of service_code (12 raw values -> 11 categories)
    3. light cleaning, then drop reports whose light-cleaned text is empty (only phrases)
    4. strong cleaning. Reports that are empty only here are kept, so that both
       topic methods work on the same set of reports
    """
    n_raw = len(df)
    df = df.dropna(subset=[text_column]).copy()
    logger.info("Reports without text removed: %d of %d", n_raw - len(df), n_raw)

    df["requested_datetime"] = pd.to_datetime(df["requested_datetime"])
    df["service_code"] = df["service_code"].map(fix_encoding)

    df["text_light"] = df[text_column].map(
        lambda text: light_clean(text, pre_cfg["phrases"], pre_cfg["email_placeholder"])
    )
    is_empty = df["text_light"] == ""
    logger.info("Reports empty after light cleaning removed: %d", is_empty.sum())
    df = df[~is_empty]

    logger.info("Strong cleaning of %d reports with spaCy ...", len(df))
    df["text_strong"] = strong_clean(
        df["text_light"],
        nlp,
        pre_cfg["keep_stopwords"],
        pre_cfg["spacy_batch_size"],
        pre_cfg.get("lemma_fixes"),
        pre_cfg.get("synonyms"),
    )
    logger.info("Reports empty only after strong cleaning (kept): %d", (df["text_strong"] == "").sum())

    return df.drop(columns=[text_column]).reset_index(drop=True)


def main() -> int:
    cfg = load_config()
    data_cfg, pre_cfg = cfg["data"], cfg["preprocessing"]

    raw = load_raw(project_path(data_cfg["raw_file"]), data_cfg["text_column"])
    nlp = load_nlp(pre_cfg["spacy_model"])
    processed = prepare(raw, data_cfg["text_column"], pre_cfg, nlp)

    target = project_path(data_cfg["processed_file"])
    target.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(target, index=False)
    logger.info("Saved %d reports to %s", len(processed), target)

    preview = processed.sample(n=min(data_cfg["preview_size"], len(processed)), random_state=cfg["seed"])
    preview.to_csv(project_path(data_cfg["preview_file"]), index=False, encoding="utf-8")
    logger.info("Saved preview (%d rows) to %s", len(preview), data_cfg["preview_file"])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
