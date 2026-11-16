"""Rule-based language tagging of review text.

Labels:

- ``arabic``: Arabic-script words only, or Arabic as the clear majority.
- ``english``: Latin-script words that are not Arabizi.
- ``arabizi``: Arabic written in Latin script, identified by digit-in-word
  spellings (``7lw``, ``ya3ni``) and a configurable word list.
- ``mixed``: both scripts, with the minority script at or above
  ``mixed_min_share`` of the words.
- ``emoji_only``: emoji and no words.
- ``other``: no Arabic or Latin words and no emoji.

Brand names are excluded from word counts unless they are the only words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.clean.normalize import extract_emojis, normalize_text, strip_emojis

ARABIC_WORD = re.compile(r"[ء-يٱ-ۓ]+")
LATIN_WORD = re.compile(r"[a-z0-9']*[a-z][a-z0-9']*")
ARABIZI_DIGIT = re.compile(r"[a-z][2356789][a-z]|^[2357][a-z]{2,}")
ENGLISH_NUMERIC = re.compile(
    r"^\d+(st|nd|rd|th|k|x|s|mo|mos|mnth|mnths|month|months|day|days|hr|hrs|min|mins|sar|sr|aed|am|pm|g|gb)$"
)

LABELS = ("arabic", "english", "mixed", "arabizi", "emoji_only", "other")


@dataclass(frozen=True)
class LanguageRules:
    """Thresholds and word lists used by :func:`detect_language`."""

    mixed_min_share: float
    arabizi_min_tokens: int
    arabizi_words: frozenset[str]
    brand_terms: frozenset[str]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> LanguageRules:
        """Build rules from the ``cleaning.language`` config section.

        Args:
            cfg: The ``cleaning.language`` mapping.

        Returns:
            A :class:`LanguageRules` instance with brand terms normalized.
        """
        return cls(
            mixed_min_share=float(cfg["mixed_min_share"]),
            arabizi_min_tokens=int(cfg["arabizi_min_tokens"]),
            arabizi_words=frozenset(w.lower() for w in cfg["arabizi_words"]),
            brand_terms=frozenset(normalize_text(t) for t in cfg["brand_terms"]),
        )


def is_arabizi_token(token: str, words: frozenset[str]) -> bool:
    """Check whether a Latin-script token is Arabizi.

    Args:
        token: Lowercase Latin-script token.
        words: Known Arabizi words.

    Returns:
        True if the token is a known Arabizi word or uses Arabizi digits.
    """
    if token in words:
        return True
    return bool(ARABIZI_DIGIT.search(token)) and not ENGLISH_NUMERIC.match(token)


def detect_language(text: str | None, rules: LanguageRules) -> str:
    """Assign a language label to a review.

    Args:
        text: Review text, raw or normalized.
        rules: Tagging thresholds and word lists.

    Returns:
        One of :data:`LABELS`.
    """
    norm = normalize_text(text)
    words = strip_emojis(norm)
    arabic = ARABIC_WORD.findall(words)
    latin_all = LATIN_WORD.findall(words)
    latin = [w for w in latin_all if len(w) >= 2]

    if not arabic and not latin:
        if len(latin_all) >= 2:  # dotted spellings such as "F.A.S.T"
            return "english"
        return "emoji_only" if extract_emojis(norm) else "other"

    arabic_nb = [w for w in arabic if w not in rules.brand_terms]
    latin_nb = [w for w in latin if w not in rules.brand_terms]
    if arabic_nb or latin_nb:
        arabic, latin = arabic_nb, latin_nb

    n_ar, n_lat = len(arabic), len(latin)
    if n_ar and n_lat and min(n_ar, n_lat) / (n_ar + n_lat) >= rules.mixed_min_share:
        return "mixed"
    if n_ar >= n_lat:
        return "arabic"
    n_arabizi = sum(is_arabizi_token(w, rules.arabizi_words) for w in latin)
    return "arabizi" if n_arabizi >= rules.arabizi_min_tokens else "english"
