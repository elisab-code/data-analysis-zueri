"""Tests for src/topics/states.py and the state patterns in the config."""

import numpy as np
import pytest

from src.config import load_config
from src.topics.states import state_markers, topic_states

PATTERNS = load_config()["states"]


@pytest.mark.parametrize(
    ("text", "state"),
    [
        ("velo stehen seit 3 woche", "duration"),
        ("seit mehrere monat", "duration"),
        ("seit ca. 2 jahr", "duration"),
        ("lampe immer noch defekt", "duration"),
        ("abfall immer wieder", "repetition"),
        ("schon wieder graffiti", "repetition"),
        ("regelmässig abfall", "repetition"),
        ("sehr gefährlich kreuzung", "danger"),
        ("unfallgefahr schulweg", "danger"),
        ("gefährden kind", "danger"),
    ],
)
def test_state_is_found(text, state):
    assert state_markers([text], PATTERNS).loc[0, state]


@pytest.mark.parametrize(
    ("text", "state"),
    [
        ("lampe defekt", "duration"),
        ("seit gestern", "duration"),  # no time unit: not counted as duration
        ("wieder voll", "repetition"),  # "wieder" alone is too vague
        ("gefahrlos queren", "danger"),
        ("sicherheitslinie fehlen", "danger"),  # a road marking, not a danger
    ],
)
def test_state_is_not_found(text, state):
    assert not state_markers([text], PATTERNS).loc[0, state]


def test_state_markers_shape():
    markers = state_markers(["abfall", "", "seit 2 woche immer wieder"], PATTERNS)
    assert list(markers.columns) == list(PATTERNS)
    assert len(markers) == 3
    assert markers.dtypes.eq(bool).all()
    assert not markers.loc[1].any()  # empty text


def test_topic_states_gives_share_per_topic():
    markers = state_markers(["seit 2 woche", "abfall", "abfall", "gefährlich"], PATTERNS)
    shares = topic_states(np.array([0, 0, 1, -1]), markers).set_index("topic")
    assert shares.loc[0, "duration"] == pytest.approx(0.5)
    assert shares.loc[1, "duration"] == 0
    assert shares.loc[-1, "danger"] == 1
