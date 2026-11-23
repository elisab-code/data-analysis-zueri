"""Tests for src/preprocessing/strong.py (uses the real spaCy model)."""

import pytest

from src.preprocessing.strong import keep_token, load_nlp, strong_clean

KEEP_WORDS = ["nicht", "kein", "seit"]
BATCH_SIZE = 2


@pytest.fixture(scope="module")
def nlp():
    # Loading the model takes a few seconds, so it is shared by all tests in this file.
    return load_nlp("de_core_news_sm")


def clean_one(text, nlp, keep_words=KEEP_WORDS):
    return strong_clean([text], nlp, keep_words, BATCH_SIZE)[0]


def test_load_nlp_disables_unneeded_components(nlp):
    assert "lemmatizer" in nlp.pipe_names
    assert "parser" not in nlp.pipe_names
    assert "ner" not in nlp.pipe_names


def test_lemmatizes_and_lowercases(nlp):
    assert clean_one("Die Lampen leuchten nicht.", nlp) == "lampe leuchten nicht"


def test_removes_stopwords_and_punctuation(nlp):
    result = clean_one("Velos stehen auf dem Trottoir!", nlp).split()
    assert "auf" not in result
    assert "der" not in result
    assert "!" not in result


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("Tempo 30 wird nicht eingehalten", "30"),
        ("Tram 4 Haltestelle", "4"),
        ("Steht hier seit dem 10.07.2024", "10.07.2024"),
    ],
)
def test_keeps_numbers(nlp, text, number):
    assert number in clean_one(text, nlp).split()


@pytest.mark.parametrize("compound", ["e-scooter", "tempo-30-zone", "led-strassenlampe"])
def test_keeps_hyphenated_compounds(nlp, compound):
    text = "E-Scooter in der Tempo-30-Zone, LED-Strassenlampen blenden"
    assert compound in clean_one(text, nlp).split()


def test_keeps_listed_stopwords(nlp):
    result = clean_one("Seit Wochen kein Abholtermin", nlp).split()
    assert "seit" in result
    assert "kein" in result


def test_removes_stopwords_not_in_keep_words(nlp):
    assert "nicht" not in clean_one("Die Lampe leuchtet nicht", nlp, keep_words=[]).split()


def test_keep_token_single_tokens(nlp):
    doc = nlp("Die Lampe , 30")
    keep = frozenset(KEEP_WORDS)
    assert [keep_token(token, keep) for token in doc] == [False, True, False, True]


def test_keeps_order_and_length(nlp):
    texts = ["Abfall", "", "Lampe defekt"]
    result = strong_clean(texts, nlp, KEEP_WORDS, BATCH_SIZE)
    assert len(result) == 3
    assert result[0] == "abfall"
    assert result[1] == ""
