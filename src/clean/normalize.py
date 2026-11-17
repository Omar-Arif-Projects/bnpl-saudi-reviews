"""Text normalization for Arabic and English reviews, and emoji extraction.

``normalize_text`` applies, in order:

1. Unicode NFKC normalization (folds stylized Latin letters and Arabic
   presentation forms to their standard code points).
2. Removal of invisible formatting characters. The zero-width joiner (U+200D)
   is preserved because it is part of multi-codepoint emoji.
3. Arabic-Indic and Extended Arabic-Indic digits to ASCII digits.
4. Removal of Arabic diacritics and tatweel.
5. Letter unification: آ أ إ ٱ -> ا, ة -> ه, ى -> ي.
6. Character runs capped at ``repeat_max``.
7. Lowercasing and whitespace collapsing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import emoji

INVISIBLE = re.compile("[​‌‎‏‪-‮⁦-⁩﻿]")
DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ALEF_VARIANTS = re.compile("[آأإٱ]")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)
LETTER_MAP = str.maketrans({"ة": "ه", "ى": "ي"})
WHITESPACE = re.compile(r"\s+")


def collapse_repeats(text: str, repeat_max: int = 2) -> str:
    """Shorten runs of a repeated character to ``repeat_max`` characters.

    Args:
        text: Input text.
        repeat_max: Maximum run length to keep. Must be at least 1.

    Returns:
        Text with long character runs shortened.

    Raises:
        ValueError: If ``repeat_max`` is less than 1.
    """
    if repeat_max < 1:
        raise ValueError("repeat_max must be >= 1")
    pattern = re.compile(r"(.)\1{%d,}" % repeat_max, flags=re.DOTALL)
    return pattern.sub(lambda m: m.group(1) * repeat_max, text)


def normalize_arabic(text: str) -> str:
    """Remove diacritics and tatweel and unify Arabic letter variants.

    Args:
        text: Input text. Non-Arabic characters are left unchanged.

    Returns:
        Normalized text.
    """
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = ALEF_VARIANTS.sub("ا", text)
    return text.translate(LETTER_MAP)


def normalize_text(text: str | None, repeat_max: int = 2) -> str:
    """Normalize a review for tokenization and modeling.

    Args:
        text: Raw review text. ``None`` is treated as an empty string.
        repeat_max: Maximum run length of a repeated character.

    Returns:
        Normalized text, possibly empty.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE.sub("", text).translate(DIGITS)
    text = normalize_arabic(text)
    text = collapse_repeats(text, repeat_max)
    return WHITESPACE.sub(" ", text.lower()).strip()


def normalize_batch(texts: Iterable[str | None], repeat_max: int = 2) -> list[str]:
    """Apply :func:`normalize_text` to a sequence of texts.

    Args:
        texts: Raw review texts.
        repeat_max: Maximum run length of a repeated character.

    Returns:
        List of normalized texts.
    """
    return [normalize_text(t, repeat_max) for t in texts]


def extract_emojis(text: str | None) -> list[str]:
    """Return the emoji in a text in order of appearance.

    Multi-codepoint emoji (skin-tone modifiers, variation selectors, ZWJ
    sequences) are returned as single items.

    Args:
        text: Input text.

    Returns:
        List of emoji strings.
    """
    if not text:
        return []
    return [match["emoji"] for match in emoji.emoji_list(text)]


def strip_emojis(text: str) -> str:
    """Remove all emoji from a text.

    Args:
        text: Input text.

    Returns:
        Text without emoji. Whitespace is not collapsed.
    """
    return emoji.replace_emoji(text, replace="")
