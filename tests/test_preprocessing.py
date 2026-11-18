"""Tests for src/preprocessing/clean.py (expected behaviour)."""

import pytest

from src.preprocessing.clean import fix_encoding, remove_html


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
