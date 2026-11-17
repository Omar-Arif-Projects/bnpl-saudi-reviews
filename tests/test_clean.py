"""Tests for Arabic normalization, emoji extraction and language tagging."""

from __future__ import annotations

import pandas as pd
import pytest

from src.clean.language import LanguageRules, detect_language, has_other_script
from src.clean.normalize import collapse_repeats, extract_emojis, normalize_text
from src.clean.pipeline import RemovalLog
from src.config import load_config


@pytest.fixture(scope="module")
def rules() -> LanguageRules:
    """Language rules loaded from the project config."""
    return LanguageRules.from_config(load_config()["cleaning"]["language"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("أفضل إفضل آخر", "افضل افضل اخر"),           # alef variants
        ("تجربة", "تجربه"),                            # taa marbuta
        ("على", "علي"),                                # alef maqsura
        ("ممـــتاز", "ممتاز"),                          # tatweel
        ("شُكْرًا", "شكرا"),                             # diacritics
        ("حلووووو", "حلوو"),                            # character runs
        ("GOOOOD   App", "good app"),                  # case, whitespace, runs
        ("١٠٠ ٪", "100 ٪"),                             # Arabic-Indic digits
        ("١٠٠٠ ريال", "1000 ريال"),                      # digit runs are not capped
        ("5000 SAR!!!!", "5000 sar!!"),                 # symbol runs are
        ("𝓣𝓱𝓪𝓷𝓴𝓼", "thanks"),                           # NFKC
        ("‏شكرا‎", "شكرا"),                   # bidi control characters
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_text(raw: str | None, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_collapse_repeats_rejects_zero() -> None:
    with pytest.raises(ValueError):
        collapse_repeats("aaa", 0)


def test_extract_emojis_keeps_sequences() -> None:
    assert extract_emojis("حلو 👍🏻 ❤️ 😍😍") == ["👍🏻", "❤️", "😍", "😍"]
    assert extract_emojis("no emoji") == []
    assert extract_emojis(None) == []


def test_normalize_keeps_emoji_joiner() -> None:
    family = "👨‍👩‍👧"
    assert extract_emojis(normalize_text(family)) == [family]


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("ممتاز من باب التيسير والتسهيل", "arabic"),
        ("شكرا Tabby", "arabic"),                       # brand term excluded
        ("Tabby", "english"),                           # brand term only
        ("great app, easy to use", "english"),
        ("I liked it mashallah", "english"),            # loan phrase
        ("2nd payment failed after 3x tries, limit 5k", "english"),
        ("التطبيق very slow", "mixed"),
        ("xممتاز", "arabic"),                           # single Latin letter
        ("el tatbeeq 7lw mara", "arabizi"),
        ("ya3ni zain bas", "arabizi"),
        ("F.A.S.T", "english"),                         # dotted abbreviation
        ("👍👍👍", "emoji_only"),
        ("...", "other"),
        ("ভালো", "other"),                              # unsupported script
    ],
)
def test_detect_language(text: str, label: str, rules: LanguageRules) -> None:
    assert detect_language(text, rules) == label


def test_has_other_script() -> None:
    assert has_other_script("ভালো")
    assert not has_other_script("ممتاز good 👍")
    assert not has_other_script("ℂ")
    assert not has_other_script(None)


def test_removal_log_counts() -> None:
    before = pd.DataFrame({"x": range(5)})
    log = RemovalLog(len(before))
    after = log.record("drop_odd", "odd numbers", before, before[before["x"] % 2 == 0])
    frame = log.to_frame()
    assert len(after) == 3
    assert frame.iloc[-1][["removed", "rows_after"]].tolist() == [2, 3]
