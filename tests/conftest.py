"""Shared pytest fixtures."""

import pytest

from src.preprocessing.strong import load_nlp


@pytest.fixture(scope="session")
def nlp():
    # Loading the spaCy model takes a few seconds, so it is shared by all tests.
    return load_nlp("de_core_news_sm")
