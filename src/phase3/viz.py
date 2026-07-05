"""Shared win-probability plotting.

`plot_winprob` (matplotlib) renders the static GIF; `plotly_winprob` renders the
interactive dark-theme chart in the Streamlit dashboard. Colours are shared so the
chart, shading, and feed all agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUARTER_MARKS = [12, 24, 36, 48]  # minutes elapsed at the end of each quarter

# Cohesive palette (home vs away) reused by the chart shading and the feed.
HOME_COLOR = "#4c9be8"   # blue
AWAY_COLOR = "#e8794c"   # orange
LINE_COLOR = "#f5f5f5"
MARKER_COLOR = "#ffd24c"  # current-moment marker


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


def _clock(period: int, secs: float) -> str:
    m, s = int(secs // 60), int(round(secs % 60))
    return f"{'OT' if period > 4 else f'Q{int(period)}'} {m}:{s:02d}"


def plotly_winprob(curve: pd.DataFrame, upto: int, home: str, away: str):
    """Interactive dark-theme win-prob chart revealed up to event `upto`.

    Hover shows clock / score / play description; the current moment is a
    prominent marker. Plotly is imported lazily so the GIF path never needs it.
    """
    import plotly.graph_objects as go

    k = max(1, min(int(upto), len(curve)))
    seg = curve.iloc[:k]
    x = seg["minutes_elapsed"].to_numpy()
    y = seg["win_prob"].to_numpy()
    descs = [(d[:48] + "…") if len(d) > 48 else d for d in seg["description"]]
    custom = np.column_stack([
        [_clock(p, s) for p, s in zip(seg["period"], seg["secs_left_period"])],
        seg["score_home"].astype(int).astype(str),
        seg["score_away"].astype(int).astype(str),
        descs,
    ])

    fig = go.Figure()
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor=HOME_COLOR, opacity=0.12, line_width=0)
    fig.add_hrect(y0=0.0, y1=0.5, fillcolor=AWAY_COLOR, opacity=0.12, line_width=0)
    fig.add_hline(y=0.5, line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dash"))
    for m in QUARTER_MARKS:
        fig.add_vline(x=m, line=dict(color="rgba(255,255,255,0.12)", width=1))

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=LINE_COLOR, width=2.5),
        customdata=custom,
        hovertemplate=(f"<b>%{{customdata[0]}}</b>   {away} %{{customdata[2]}} – "
                       f"%{{customdata[1]}} {home}<br>P({home} win): %{{y:.0%}}"
                       f"<br>%{{customdata[3]}}<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[y[-1]], mode="markers",
        marker=dict(color=MARKER_COLOR, size=15, line=dict(color="#111", width=2)),
        hoverinfo="skip", showlegend=False,
    ))

    xmax = max(48.0, float(np.ceil(x.max())))
    fig.update_layout(
        template="plotly_dark", height=440, showlegend=False,
        margin=dict(l=50, r=20, t=10, b=40),
        xaxis=dict(title="Game minutes elapsed", range=[0, xmax], showgrid=False),
        yaxis=dict(title=f"P({home} win)", range=[0, 1], tickformat=".0%", showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#1a1f2b"),
    )
    return fig
