"""Text cleaning steps shared by both text versions (light and strong).

Each function takes and returns a single string. Missing values (NaN) are
handled by the caller on DataFrame level, not here.
"""

import re

import ftfy

# Matches line-break tags only: <br/>, <br>, <br />, <BR/>.
# Other text in angle brackets (e.g. "<Schlagloch>") is written by citizens and kept.
LINE_BREAK_TAG = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)


def fix_encoding(text: str) -> str:
    """Repair mojibake (UTF-8 bytes that were decoded as Windows-1252).

    Examples: "MÃ¼ll" -> "Müll". Correct text stays unchanged.
    """
    return ftfy.fix_encoding(text)


def remove_html(text: str) -> str:
    """Replace HTML line breaks with a single space.

    The only real tag in `detail` is the line break ("<br/>").
    """
    return LINE_BREAK_TAG.sub(" ", text)
