"""Arabic-aware text normalization and emoji extraction.

Normalization makes different spellings of the same word identical so that
models and word counts treat them as one token. The original review text is
never modified; the normalized form goes into a separate column.

Steps in :func:`normalize_text` (in order):
0. Unicode NFKC: turns "fancy font" letters (𝓣𝓱𝓪𝓷𝓴𝓼) into plain ones and
   Arabic presentation forms (ﻻ) into standard letters.
1. Remove invisible direction / zero-width marks (the zero-width joiner, which
   holds multi-part emoji together, is kept).
2. Arabic-Indic and Persian digits -> 0-9.
3. Remove diacritics (tashkeel, superscript alef) and tatweel (ـ).
4. Unify letter variants: أ إ آ ٱ -> ا, ة -> ه, ى -> ي. (Trade-off: على "on"
   and علي "Ali" become the same word; this is standard for Arabic NLP.)
5. Cap runs of the same character at ``repeat_max`` (حلووووو -> حلوو).
   Two are kept rather than one so emphasis stays visible.
6. Lowercase Latin letters and collapse whitespace.

Emoji are kept in the normalized text and also extracted by
:func:`extract_emojis`, so they can be used as a feature.
"""

from __future__ import annotations

import re
import unicodedata

import emoji

# Invisible marks: LRM/RLM, zero-width space/non-joiner, bidi embeddings and
# isolates, BOM. U+200D (zero-width joiner) is NOT removed: emoji need it.
INVISIBLE = re.compile("[​‌‎‏‪-‮⁦-⁩﻿]")
DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ALEF_VARIANTS = re.compile("[آأإٱ]")  # آ أ إ ٱ
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)
LETTER_MAP = str.maketrans({"ة": "ه", "ى": "ي"})  # ة->ه, ى->ي
WHITESPACE = re.compile(r"\s+")


def collapse_repeats(text: str, repeat_max: int = 2) -> str:
    """Cap runs of any repeated character at ``repeat_max``.

    Args:
        text: Input text.
        repeat_max: Longest run to keep (must be >= 1).

    Returns:
        Text with long runs shortened, e.g. ``"حلووووو"`` -> ``"حلوو"`` for 2.
    """
    if repeat_max < 1:
        raise ValueError("repeat_max must be >= 1")
    pattern = re.compile(r"(.)\1{%d,}" % repeat_max, flags=re.DOTALL)
    return pattern.sub(lambda m: m.group(1) * repeat_max, text)


def normalize_arabic(text: str) -> str:
    """Apply the Arabic-specific steps: diacritics, tatweel, letter variants.

    Args:
        text: Input text (any script; non-Arabic characters are untouched).

    Returns:
        Text with Arabic orthographic variation removed.
    """
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = ALEF_VARIANTS.sub("ا", text)
    return text.translate(LETTER_MAP)


def normalize_text(text: str | None, repeat_max: int = 2) -> str:
    """Fully normalize one review for counting and modeling.

    Args:
        text: Raw review text (``None`` is treated as empty).
        repeat_max: Longest run of a repeated character to keep.

    Returns:
        Normalized text (possibly empty).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE.sub("", text).translate(DIGITS)
    text = normalize_arabic(text)
    text = collapse_repeats(text, repeat_max)
    return WHITESPACE.sub(" ", text.lower()).strip()


def extract_emojis(text: str | None) -> list[str]:
    """Return the emoji in a text, in order, as whole sequences.

    Multi-part emoji (skin tones, ❤️ with its variation selector, ZWJ families)
    are returned as one item.

    Args:
        text: Review text.

    Returns:
        List of emoji strings (empty if none).
    """
    if not text:
        return []
    return [match["emoji"] for match in emoji.emoji_list(text)]


def strip_emojis(text: str) -> str:
    """Remove all emoji from a text.

    Args:
        text: Input text.

    Returns:
        Text with emoji removed (whitespace is not re-collapsed).
    """
    return emoji.replace_emoji(text, replace="")
