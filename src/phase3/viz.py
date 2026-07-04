"""Shared win-probability plotting, used by both the GIF and the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd

QUARTER_MARKS = [12, 24, 36, 48]  # minutes elapsed at the end of each quarter


def plot_winprob(ax, curve: pd.DataFrame, upto: int, home: str, away: str):
    """Draw the win-prob curve revealed up to event `upto` (1-based)."""
    ax.clear()
    x = curve["minutes_elapsed"].to_numpy()
    y = curve["win_prob"].to_numpy()
    k = max(1, min(upto, len(curve)))

    # Shade the home-leading (blue, >0.5) and away-leading (red, <0.5) regions.
    ax.axhspan(0.5, 1.0, color="#e8f0fe", zorder=0)
    ax.axhspan(0.0, 0.5, color="#fdecea", zorder=0)
    ax.axhline(0.5, color="gray", lw=1, ls="--", zorder=1)
    for m in QUARTER_MARKS:
        ax.axvline(m, color="white", lw=1.5, zorder=1)

    ax.plot(x[:k], y[:k], color="#111111", lw=2.2, zorder=3)
    ax.scatter([x[k - 1]], [y[k - 1]], color="crimson", s=45, zorder=4)

    xmax = max(48, float(np.ceil(x.max())))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Game minutes elapsed")
    ax.set_ylabel(f"P({home} win)")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    row = curve.iloc[k - 1]
    period = int(row["period"])
    clock = "OT" if period > 4 else f"Q{period}"
    ax.set_title(
        f"{away} @ {home}   |   {home} {int(row['score_home'])} - "
        f"{int(row['score_away'])} {away}   |   {clock}   "
        f"P({home} win) = {y[k - 1]:.0%}",
        fontsize=11,
    )
    return ax
