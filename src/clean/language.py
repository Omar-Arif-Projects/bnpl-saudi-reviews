"""Tag each review's language from its text: Arabic, English, mixed, Arabizi.

The Play Store ``lang_query`` bucket is not reliable (the ``en`` query returns
some Arabic reviews and vice versa), so language is decided from the words.

Labels:
- ``arabic``: Arabic-script words only (or nearly only).
- ``english``: Latin-script words, not Arabizi.
- ``arabizi``: Arabic written in Latin letters, often with digits for sounds
  (``7lw`` = حلو, ``ya3ni`` = يعني). Detected by digit-in-word patterns and a
  small word list from ``config.yaml``; needs ``arabizi_min_tokens`` hits so one
  stray "inshallah" in an English review does not count.
- ``mixed``: both scripts, with the smaller one at least ``mixed_min_share`` of
  the words (e.g. "التطبيق very slow").
- ``emoji_only``: no words, only emoji.
- ``other``: no Arabic or Latin words and no emoji (digits, punctuation, other
  scripts such as Bengali). Removed by the pipeline.

Brand names ("tabby", "تمارا", ...) are ignored when counting, so "شكرا Tabby"
is Arabic, not mixed. If a review is only a brand name, they are counted.

This is a rule-based heuristic, not a trained detector. Its output is checked by
eye in the EDA notebook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.clean.normalize import extract_emojis, normalize_text, strip_emojis

ARABIC_WORD = re.compile(r"[ء-يٱ-ۓ]+")
LATIN_WORD = re.compile(r"[a-z0-9']*[a-z][a-z0-9']*")
# Arabizi digit use: a sound digit between letters (ya3ni, m3a) or leading (7lw).
ARABIZI_DIGIT = re.compile(r"[a-z][2356789][a-z]|^[2357][a-z]{2,}")
# English tokens that also mix digits and letters: 2nd, 3x, 5k, 2mnths, 48hrs.
ENGLISH_NUMERIC = re.compile(
    r"^\d+(st|nd|rd|th|k|x|s|mo|mos|mnth|mnths|month|months|day|days|hr|hrs|min|mins|sar|sr|aed|am|pm|g|gb)$"
)

LABELS = ("arabic", "english", "mixed", "arabizi", "emoji_only", "other")


@dataclass(frozen=True)
class LanguageRules:
    """Thresholds and word lists for :func:`detect_language` (from config)."""

    mixed_min_share: float
    arabizi_min_tokens: int
    arabizi_words: frozenset[str]
    brand_terms: frozenset[str]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> LanguageRules:
        """Build rules from the ``cleaning.language`` section of config.yaml.

        Args:
            cfg: The ``cleaning.language`` mapping.

        Returns:
            A :class:`LanguageRules` instance. Brand terms are normalized the
            same way as review text so they match after normalization.
        """
        return cls(
            mixed_min_share=float(cfg["mixed_min_share"]),
            arabizi_min_tokens=int(cfg["arabizi_min_tokens"]),
            arabizi_words=frozenset(w.lower() for w in cfg["arabizi_words"]),
            brand_terms=frozenset(normalize_text(t) for t in cfg["brand_terms"]),
        )


def is_arabizi_token(token: str, words: frozenset[str]) -> bool:
    """Return True if a Latin token looks like Arabizi.

    Args:
        token: Lowercase Latin token.
        words: Known Arabizi words.

    Returns:
        Whether the token is in the word list or uses Arabizi digits.
    """
    if token in words:
        return True
    return bool(ARABIZI_DIGIT.search(token)) and not ENGLISH_NUMERIC.match(token)


def detect_language(text: str | None, rules: LanguageRules) -> str:
    """Tag one review with a language label.

    Args:
        text: Raw or normalized review text.
        rules: Thresholds and word lists.

    Returns:
        One of :data:`LABELS`.
    """
    norm = normalize_text(text)
    words = strip_emojis(norm)
    arabic = ARABIC_WORD.findall(words)
    latin_all = LATIN_WORD.findall(words)
    # Single Latin letters ("x", stray keyboard hits) are not words...
    latin = [w for w in latin_all if len(w) >= 2]

    if not arabic and not latin:
        if len(latin_all) >= 2:
            return "english"  # ...unless the review is spelled out: "F.A.S.T", "v.g"
        return "emoji_only" if extract_emojis(norm) else "other"

    # Ignore brand names when counting, unless they are all there is.
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
