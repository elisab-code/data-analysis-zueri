"""Tests for src/preprocessing/pipeline.py on a small hand-made DataFrame."""

import pandas as pd
import pytest

from src.preprocessing.pipeline import prepare

PRE_CFG = {
    "phrases": ["guten tag", "danke"],
    "email_placeholder": "",
    "keep_stopwords": ["nicht"],
    "spacy_batch_size": 2,
}


@pytest.fixture
def raw():
    return pd.DataFrame(
        {
            "service_request_id": [1, 2, 3, 4, 5],
            "requested_datetime": ["2020-01-01T10:00:00"] * 5,
            "service_code": ["GrÃ¼nflÃ¤chen/SpielplÃ¤tze", "Abfall/Sammelstelle", "Graffiti", "Allgemein", "Graffiti"],
            "interface_used": ["iOS"] * 5,
            "detail": [
                "Guten Tag <br/> Die Lampe leuchtet nicht. Danke",  # normal report
                None,                                                # no text
                "Danke!",                                            # only a phrase
                "Hier ist es.",                                      # only stopwords
                "Schrottvelo",                                       # one word
            ],
        }
    )


@pytest.fixture
def processed(raw, nlp):
    return prepare(raw, "detail", PRE_CFG, nlp)


def test_removes_missing_and_phrase_only_reports(processed):
    assert processed["service_request_id"].tolist() == [1, 4, 5]


def test_keeps_reports_empty_only_after_strong_cleaning(processed):
    row = processed.set_index("service_request_id").loc[4]
    assert row["text_light"] == "Hier ist es."
    assert row["text_strong"] == ""


def test_output_columns(processed):
    assert list(processed.columns) == [
        "service_request_id", "requested_datetime", "service_code",
        "interface_used", "text_light", "text_strong",
    ]


def test_repairs_service_code(processed):
    assert processed.loc[0, "service_code"] == "Grünflächen/Spielplätze"


def test_texts_are_cleaned(processed):
    assert processed.loc[0, "text_light"] == "Die Lampe leuchtet nicht."
    assert processed.loc[0, "text_strong"] == "lampe leuchten nicht"


def test_datetime_is_parsed(processed):
    assert pd.api.types.is_datetime64_any_dtype(processed["requested_datetime"])


def test_does_not_modify_input(raw, nlp):
    before = raw.copy()
    prepare(raw, "detail", PRE_CFG, nlp)
    pd.testing.assert_frame_equal(raw, before)
