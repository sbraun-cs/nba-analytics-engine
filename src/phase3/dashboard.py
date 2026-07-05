"""Phase 3 - Streamlit win-probability replay dashboard.

Replays a historical game with a live-updating (Plotly) win-probability curve, a
play-by-play feed, and running leading scorers. Needs the cached play-by-play in
data/ (local only).

Run:  python -m streamlit run src/phase3/dashboard.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

# Make the project root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.phase3.baseline import (
    SCHEMA_VERSION, TEST_SEASON, find_demo_game, fmt_clock, game_curve,
    game_meta, game_teams, leading_scorers, sampled_split, train_model,
)
from src.phase3.viz import AWAY_COLOR, HOME_COLOR, plotly_winprob

st.set_page_config(page_title="NBA Win Probability Replay", layout="wide")

# schema_version threads into every cache key so a SCHEMA_VERSION bump invalidates
# any stale cache a long-running session would otherwise keep serving.
@st.cache_resource
def get_model(schema_version: int = SCHEMA_VERSION):
    return train_model()


@st.cache_data
def get_demo_game(schema_version: int = SCHEMA_VERSION):
    return find_demo_game(get_model())


@st.cache_data
def get_meta(schema_version: int = SCHEMA_VERSION):
    return game_meta(TEST_SEASON)


@st.cache_data
def get_curve(game_id: str, schema_version: int = SCHEMA_VERSION):
    return game_curve(get_model(), game_id), game_teams(game_id)


model = get_model()
demo_id, demo_swing = get_demo_game()
meta = get_meta()
_, test_ids = sampled_split()

st.title("🏀 NBA Win Probability — Game Replay")

# --- game picker (spoiler-free by default) -----------------------------------
hide_final = st.sidebar.toggle("Spoiler-free (hide final scores)", value=True)


def label_for(game_id: str) -> str:
    m = meta.get(game_id)
    base = f"{m['date']} — {m['away']} @ {m['home']}" if m else game_id
    if m and not hide_final:
        base += f" ({m['away_pts']}-{m['home_pts']})"
    if game_id == demo_id:
        base += "   ⭐ biggest Q4 swing"
    return base


options = [demo_id] + [g for g in test_ids if g != demo_id]
game_id = st.selectbox("Game", options=options, format_func=label_for)

curve, (home, away) = get_curve(game_id)
n = len(curve)
m = meta.get(game_id, {})

# --- header block ------------------------------------------------------------
st.markdown(
    f"## <span style='color:{AWAY_COLOR}'>{away}</span> "
    f"<span style='color:#888'>@</span> "
    f"<span style='color:{HOME_COLOR}'>{home}</span>",
    unsafe_allow_html=True,
)
st.caption(f"{m.get('date', '')} · win-probability replay · Phase 3 logistic + prior model")

# Full-width KPI row so long values like "HOU +27" never truncate in a narrow column.
metrics = st.empty()

left, right = st.columns([3, 2])
with right:
    idx = st.slider("Replay to event", 1, n, n)
    play = st.button("▶ Play from start")

chart = left.empty()
scorer = left.empty()
feed = right.empty()

# Every plotly_chart in a run needs a unique key, else the Play loop (many
# re-renders into one placeholder within a single run) raises
# StreamlitDuplicateElementId. A per-run counter guarantees uniqueness.
_chart_keys = itertools.count()


def _feed_html(k: int) -> str:
    rows = ["<div style='font-weight:700;margin-bottom:4px'>Play-by-play</div>"]
    for _, e in curve.iloc[max(0, k - 8):k].iloc[::-1].iterrows():
        scoring = e["points"] > 0
        team_color = HOME_COLOR if e["team_tricode"] == home else AWAY_COLOR
        accent = team_color if scoring else "transparent"
        bg = "rgba(255,255,255,0.05)" if scoring else "transparent"
        clock = fmt_clock(e["period"], e["secs_left_period"])
        score = ""
        if scoring:
            score = (f"<span style='float:right;font-weight:700;color:{team_color}'>"
                     f"{int(e['score_away'])}–{int(e['score_home'])}</span>")
        rows.append(
            f"<div style='border-left:3px solid {accent};background:{bg};"
            f"padding:3px 8px;margin:3px 0;border-radius:3px'>"
            f"<span style='font-family:monospace;font-size:11px;background:#2b2b2b;"
            f"color:#bbb;padding:1px 5px;border-radius:3px'>{clock}</span> "
            f"<b style='color:{team_color}'>{e['team_tricode'] or '—'}</b> "
            f"<span style='color:#ddd'>{e['description']}</span>{score}</div>"
        )
    return "".join(rows)


def render(k: int):
    chart.plotly_chart(plotly_winprob(curve, k, home, away), width="stretch",
                       key=f"winprob_{next(_chart_keys)}")

    row = curve.iloc[k - 1]
    prev = curve.iloc[k - 2] if k >= 2 else row
    d_margin = int(row["score_margin"] - prev["score_margin"])
    d_wp = (row["win_prob"] - prev["win_prob"]) * 100
    lead = int(row["score_margin"])
    lead_label = f"{home} +{lead}" if lead > 0 else (f"{away} +{-lead}" if lead < 0 else "Tied")

    with metrics.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{home} (home)", int(row["score_home"]))
        c2.metric(f"{away} (away)", int(row["score_away"]))
        c3.metric("Lead", lead_label, delta=(d_margin or None))
        c4.metric(f"P({home} win)", f"{row['win_prob']:.0%}",
                  delta=(f"{d_wp:+.1f} pts" if abs(d_wp) >= 0.05 else None))

    ls = leading_scorers(curve, k, home, away)
    top = lambda tri: f"{ls[tri][0]} {ls[tri][1]}" if ls[tri] else "—"
    scorer.markdown(f"**Leading scorers** — {home}: {top(home)} · {away}: {top(away)}")

    feed.markdown(_feed_html(k), unsafe_allow_html=True)


if play:
    step = max(1, n // 200)
    for k in range(1, n + 1, step):
        render(k)
        time.sleep(0.03)
    render(n)
else:
    render(idx)
