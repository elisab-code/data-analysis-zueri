"""State markers: how a report describes its problem, independent of the object.

Topics describe what a report is about (the object, e.g. a lamp or waste).
State markers describe how the problem is described: for how long it has
existed (duration), whether it happens again (repetition), whether it is
dangerous (danger). Crossed with the topics they show, for example, which
objects are often reported "seit Wochen".
"""

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


def compile_states(patterns: dict[str, list[str]]) -> dict[str, re.Pattern]:
    """One regular expression per state that matches if any of its patterns matches."""
    return {state: re.compile("|".join(f"(?:{p})" for p in parts)) for state, parts in patterns.items()}


def state_markers(texts: Sequence[str], patterns: dict[str, list[str]]) -> pd.DataFrame:
    """One boolean column per state and one row per text: True if the state occurs.

    A text can have several states or none. `texts` are the strongly cleaned
    texts (lowercase lemmas), so "seit 3 Wochen" is matched as "seit 3 woche".
    """
    texts = pd.Series(list(texts), dtype="string")
    return pd.DataFrame(
        {state: texts.str.contains(regex).to_numpy(dtype=bool) for state, regex in compile_states(patterns).items()}
    )


def topic_states(labels: np.ndarray, markers: pd.DataFrame) -> pd.DataFrame:
    """Share of the reports of each topic that have each state (one row per topic).

    Example: duration 0.30 for a topic means that 30 % of its reports describe
    a problem that has existed for some time.
    """
    return markers.groupby(np.asarray(labels)).mean().rename_axis("topic").reset_index()
