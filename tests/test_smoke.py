"""Smoke tests: is the environment complete and does the basic setup work?"""

import importlib

import pytest

from src.config import load_config
from src.data.download import sha256sum

REQUIRED_MODULES = [
    "pandas", "numpy", "sklearn", "ftfy", "spacy", "sentence_transformers",
    "torch", "gensim", "matplotlib", "seaborn", "yaml", "requests",
]


@pytest.mark.parametrize("module", REQUIRED_MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_spacy_german_model_loads():
    import spacy

    nlp = spacy.load("de_core_news_sm")
    lemmas = [token.lemma_ for token in nlp("Die Lampen leuchten nicht.")]
    assert "Lampe" in lemmas


def test_gensim_coherence_runs():
    """gensim must work with the installed numpy/scipy versions."""
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    texts = [["abfall", "sack", "strasse"], ["lampe", "licht", "defekt"],
             ["abfall", "strasse", "sauber"], ["licht", "lampe", "strasse"]]
    topics = [["abfall", "strasse"], ["lampe", "licht"]]
    model = CoherenceModel(topics=topics, texts=texts,
                           dictionary=Dictionary(texts), coherence="c_v")
    assert -1.0 <= model.get_coherence() <= 1.0


def test_config_has_required_keys():
    cfg = load_config()
    assert isinstance(cfg["seed"], int)
    for key in ("raw_file", "sha256", "text_column"):
        assert cfg["data"][key]


def test_sha256sum_matches_known_value(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"abc")
    assert sha256sum(sample) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
