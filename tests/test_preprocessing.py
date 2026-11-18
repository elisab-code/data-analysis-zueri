"""Tests for src/preprocessing/clean.py (expected behaviour)."""

import pytest

from src.preprocessing.clean import (
    fix_encoding,
    light_clean,
    mask_emails,
    normalize_whitespace,
    remove_html,
    remove_phrases,
)

PHRASES = ["guten tag", "vielen dank im voraus", "vielen dank", "danke", "freundliche grüsse"]


@pytest.mark.parametrize(
    ("broken", "expected"),
    [
        ("MÃ¼ll", "Müll"),
        ("GrÃ¼nflÃ¤chen/SpielplÃ¤tze", "Grünflächen/Spielplätze"),
        ("Super ðŸ˜€", "Super 😀"),
    ],
)
def test_fix_encoding_repairs_mojibake(broken, expected):
    assert fix_encoding(broken) == expected


@pytest.mark.parametrize("text", ["Müll", "Strasse", "Tempo 30", ""])
def test_fix_encoding_keeps_correct_text(text):
    assert fix_encoding(text) == text


def test_remove_html_replaces_line_break():
    assert remove_html("Lampe<br/>defekt") == "Lampe defekt"


def test_remove_html_handles_repeated_breaks():
    result = remove_html("Guten Tag <br/>  <br/> Danke")
    assert "<" not in result
    assert result.split() == ["Guten", "Tag", "Danke"]


@pytest.mark.parametrize("text", ["<Schlagloch> auf der Strasse", "Tempo < 30", "Hallo"])
def test_remove_html_keeps_other_angle_brackets(text):
    assert remove_html(text) == text


# --- light cleaning ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Lampe   defekt", "Lampe defekt"),
        ("  Lampe defekt  ", "Lampe defekt"),
        ("Lampe\n\tdefekt", "Lampe defekt"),
        ("   ", ""),
        ("", ""),
    ],
)
def test_normalize_whitespace(text, expected):
    assert normalize_whitespace(text) == expected


def test_mask_emails_replaces_address():
    assert mask_emails("Von: <max.muster@example.ch> Betreff", "EMAIL") == "Von: <EMAIL> Betreff"


@pytest.mark.parametrize("text", ["Hinweis an @stadtzuerich", "Tempo 30", "Hallo"])
def test_mask_emails_keeps_text_without_address(text):
    assert mask_emails(text, "EMAIL") == text


def test_remove_phrases_keeps_content():
    result = remove_phrases("Guten Tag, die Lampe ist defekt. Freundliche Grüsse", PHRASES)
    assert result.split() == ["die", "Lampe", "ist", "defekt."]


def test_remove_phrases_is_case_insensitive():
    assert remove_phrases("GUTEN TAG Abfall", PHRASES).split() == ["Abfall"]


def test_remove_phrases_prefers_longer_phrase():
    assert remove_phrases("Abfall. Vielen Dank im Voraus!", PHRASES).split() == ["Abfall."]


@pytest.mark.parametrize("text", ["Dankeschön für die Reparatur", "Undank ist der Welten Lohn"])
def test_remove_phrases_matches_whole_words_only(text):
    assert remove_phrases(text, PHRASES) == text


def test_light_clean_end_to_end():
    raw = "Guten Tag <br/>  <br/> Die Lampe ist defekt. <br/> Kontakt: max@example.ch <br/> Danke"
    assert light_clean(raw, PHRASES, "EMAIL") == "Die Lampe ist defekt. Kontakt: EMAIL"


def test_light_clean_repairs_encoding_first():
    assert light_clean("Guten Tag, MÃ¼ll", PHRASES, "EMAIL") == "Müll"
