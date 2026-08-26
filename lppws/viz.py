"""Small plots that make the event representation concrete.

`neuralset` already ships the general-purpose views in
:mod:`neuralset.events.viz` (``plot_events``, ``plot_study``, ``plot_overlap``).
This module adds the one picture those cannot give you: a zoom close enough to
read the individual words, with the fMRI sampling grid drawn on top.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

TR = 2.0


def plot_word_timing(
    events: pd.DataFrame,
    timeline: str | None = None,
    *,
    t0: float = 0.0,
    t1: float = 12.0,
    tr: float = TR,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw the words of one run on a time axis, over the fMRI sampling grid.

    Every word is a box spanning its own onset and duration, with the word
    written inside it. The shaded bands behind them are the fMRI samples: one
    volume every ``tr`` seconds. The point of the figure is the mismatch — many
    words land inside a single band, so the brain data cannot possibly follow
    them one by one.

    Parameters
    ----------
    events : pd.DataFrame
        Events from the study. If it holds several timelines, pass ``timeline``
        or the first one is used.
    t0, t1 : float
        Time window to show, in seconds.
    """
    if timeline is None:
        timeline = events["timeline"].iloc[0]
    words = events[(events.timeline == timeline) & (events.type == "Word")]
    window = words[(words.start >= t0) & (words.start < t1)]

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 2.6))

    # fMRI sampling grid: one shaded band per acquired volume
    n = 0
    t = t0 - (t0 % tr)
    while t < t1:
        n += 1
        ax.axvspan(t, min(t + tr, t1), color="C0", alpha=0.06 if n % 2 else 0.14, lw=0)
        ax.axvline(t, color="C0", lw=0.8, alpha=0.5)
        if t + tr <= t1:
            ax.text(t + tr / 2, 0.86, f"volume {n}", ha="center", va="center",
                    fontsize=8, color="C0")
        t += tr

    # the words themselves
    for _, w in window.iterrows():
        ax.add_patch(plt.Rectangle((w.start, 0.2), w.duration, 0.3,
                                   facecolor="crimson", alpha=0.30, edgecolor="crimson"))
        ax.text(w.start + w.duration / 2, 0.35, w.text, ha="center", va="center",
                fontsize=8, rotation=45)

    per_tr = len(window) / max((t1 - t0) / tr, 1)
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(
        f"{len(window)} words in {t1 - t0:.0f} s — about {per_tr:.1f} of them "
        f"inside every {tr:.0f} s fMRI volume"
    )
    return ax
