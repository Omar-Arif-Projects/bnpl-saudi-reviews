"""Jupyter widget for labeling the gold set.

Shows one review at a time with its app and language. The star rating is
hidden to avoid anchoring the label on it. Choosing a sentiment saves the
row with the selected topic and moves to the next unlabeled review; flag
buttons save the row as flagged. Every change is written to disk at once, so
labeling can stop and resume at any point.

Saving re-reads the file and writes only the current row's labels, so a
second open labeler does not overwrite labels saved by the first. If a save
fails (for example while a sync client locks the file), the error is shown
and the widget stays on the row so the label can be saved again.

Usage in a notebook::

    from src.label.widget import GoldLabeler
    GoldLabeler().show()                                  # gold set
    GoldLabeler(get_path("gold_relabel")).show()          # relabel sample
"""

from __future__ import annotations

import html
from pathlib import Path

import ipywidgets as w
from IPython.display import display

from src.config import get_path, load_config
from src.label.gold import LABEL_COLUMNS, is_done, load_gold, save_gold

TEXT_STYLE = (
    "font-size:18px; line-height:1.7; padding:12px 16px; border:1px solid #c3c2b7;"
    "border-radius:6px; background:#fcfcfb; color:#0b0b0b; white-space:pre-wrap; min-height:60px;"
)
ERROR_STYLE = "color:#b42318"
SENTIMENT_STYLE = {"negative": "danger", "neutral": "warning", "positive": "success"}


class GoldLabeler:
    """Interactive labeler bound to a gold-format CSV file."""

    def __init__(self, path: Path | None = None) -> None:
        """Load the file, build the widgets and show the first unlabeled row.

        Args:
            path: Gold-format CSV; defaults to ``paths.gold``.
        """
        labels = load_config()["labels"]
        self.path = path or get_path("gold")
        self.df = load_gold(self.path)
        self.classes: list[str] = labels["sentiment_classes"]
        self.flags: list[str] = labels["flags"]
        topics: dict[str, str] = labels["topics"]

        self.pos = self._next_open(0)
        self.status = w.HTML()
        self.meta = w.HTML()
        self.text = w.HTML()
        self.message = w.HTML()
        self.topic = w.ToggleButtons(
            options=[(name, key) for key, name in topics.items()], value=None,
            style={"button_width": "230px"},
        )
        self.sentiment_buttons = [
            w.Button(description=c.capitalize(), button_style=SENTIMENT_STYLE.get(c, ""))
            for c in self.classes
        ]
        for button, label in zip(self.sentiment_buttons, self.classes):
            button.on_click(lambda _, s=label: self.label(sentiment=s))
        self.flag_buttons = [w.Button(description=f.replace("_", " ").capitalize()) for f in self.flags]
        for button, flag in zip(self.flag_buttons, self.flags):
            button.on_click(lambda _, f=flag: self.label(flag=f))
        back, skip, clear = w.Button(description="◀ Back"), w.Button(description="Skip ▶"), w.Button(description="Clear")
        back.on_click(lambda _: self.move(-1))
        skip.on_click(lambda _: self.move(1))
        clear.on_click(lambda _: self.clear())

        self.box = w.VBox([
            self.status, self.meta, self.text,
            w.HTML("<b>Main topic</b>"), self.topic,
            w.HTML("<b>Sentiment</b> (saves and moves on)"), w.HBox(self.sentiment_buttons),
            w.HTML("<b>Flag</b> (excluded from evaluation)"), w.HBox(self.flag_buttons),
            w.HBox([back, skip, clear]), self.message,
        ])
        self.render()

    def _next_open(self, start: int) -> int:
        """Return the index of the first unfinished row at or after ``start``, else ``start``."""
        done = is_done(self.df).to_numpy()
        for i in range(start, len(self.df)):
            if not done[i]:
                return i
        return min(start, len(self.df) - 1)

    def _save_row(self, values: list[str]) -> bool:
        """Write the current row's labels to disk, keeping other rows as saved.

        The file is re-read first, so labels saved by another labeler are kept.
        On success the in-memory table is replaced by the saved file.

        Args:
            values: New values for :data:`LABEL_COLUMNS`, in that order.

        Returns:
            True if the row was saved; False if saving failed (the error is shown).
        """
        gid = self.df.iloc[self.pos]["gold_id"]
        try:
            fresh = load_gold(self.path)
            fresh.loc[fresh["gold_id"] == gid, LABEL_COLUMNS] = values
            save_gold(fresh, self.path)
            self.df = load_gold(self.path)
        except Exception as exc:  # e.g. the file is locked by another program
            self.message.value = (
                f'<span style="{ERROR_STYLE}">Not saved: {html.escape(str(exc))}. '
                "Click the button again to retry.</span>"
            )
            return False
        return True

    def render(self) -> None:
        """Display the current row and progress."""
        row = self.df.iloc[self.pos]
        n_done = int(is_done(self.df).sum())
        self.status.value = f"<b>{n_done} / {len(self.df)}</b> done &nbsp;·&nbsp; review {self.pos + 1}"
        state = ""
        if row["flag"]:
            state = f" &nbsp;·&nbsp; flagged: <i>{row['flag']}</i>"
        elif row["sentiment"]:
            state = f" &nbsp;·&nbsp; labeled: <i>{row['sentiment']}</i>"
        self.meta.value = f"{row['app'].capitalize()} &nbsp;·&nbsp; {row['lang']}{state}"
        body = html.escape(row["text"]) or "<i>(text removed)</i>"
        self.text.value = f'<div dir="auto" style="{TEXT_STYLE}">{body}</div>'
        self.topic.value = row["topic"] or None
        self.message.value = ""

    def label(self, sentiment: str = "", flag: str = "") -> None:
        """Save a sentiment (with the selected topic) or a flag for the current row.

        Args:
            sentiment: Sentiment class, or ``""`` when flagging.
            flag: Flag name, or ``""`` when labeling.
        """
        topic = self.topic.value or ""
        if sentiment and not topic:
            self.message.value = f'<span style="{ERROR_STYLE}">Choose a topic first.</span>'
            return
        if not self._save_row([sentiment, topic, flag]):
            return
        if is_done(self.df).all():
            self.render()
            self.message.value = "<b>All reviews are labeled.</b>"
            return
        nxt = self._next_open(self.pos + 1)
        self.pos = nxt if not is_done(self.df).iloc[nxt] else self._next_open(0)
        self.render()

    def clear(self) -> None:
        """Remove the labels of the current row."""
        if self._save_row(["", "", ""]):
            self.render()

    def move(self, step: int) -> None:
        """Move to another row without saving.

        Args:
            step: Offset from the current row.
        """
        self.pos = max(0, min(len(self.df) - 1, self.pos + step))
        self.render()

    def show(self) -> None:
        """Display the widget."""
        display(self.box)
