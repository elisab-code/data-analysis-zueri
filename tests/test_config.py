"""Consistency checks for the word lists in configs/config.yaml."""

from src.config import load_config

CFG = load_config()
PRE = CFG["preprocessing"]
EXCLUDED = {word for group in CFG["tfidf"]["exclude_words"].values() for word in group}


def test_kept_stopwords_are_excluded_from_tfidf():
    # The stopwords are kept for the state markers, not as TF-IDF terms.
    assert set(PRE["keep_stopwords"]) <= EXCLUDED


def test_synonyms_are_not_chained():
    # A target must not be mapped again, the result would depend on the order.
    targets = set(PRE["synonyms"].values())
    assert not targets & set(PRE["synonyms"])
    assert not set(PRE["lemma_fixes"].values()) & set(PRE["lemma_fixes"])


def test_lists_are_lowercase():
    # Lemmas are lowercase, so uppercase entries would never match.
    words = [*PRE["lemma_fixes"].items(), *PRE["synonyms"].items()]
    assert all(a == a.lower() and b == b.lower() for a, b in words)
    assert all(word == word.lower() for word in EXCLUDED)


def test_excluded_words_are_not_topic_targets():
    # Words that synonyms map to would be removed from TF-IDF again.
    assert not set(PRE["synonyms"].values()) & EXCLUDED
