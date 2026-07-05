"""Phase 3 - Streamlit win-probability replay dashboard.

Replays a historical game with a live-updating win-probability curve, a
play-by-play feed, and running leading scorers, using the Phase 3 model.
Needs the cached play-by-play in data/ (local only).

Run:  python -m streamlit run src/phase3/dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the project root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import streamlit as st

from src.phase3.baseline import (
    TEST_SEASON, find_demo_game, fmt_clock, game_curve, game_labels,
    game_teams, leading_scorers, sampled_split, train_model,
)
from src.phase3.viz import plot_winprob

st.set_page_config(page_title="NBA Win Probability Replay", layout="wide")


@st.cache_resource
def get_model():
    return train_model()


@st.cache_data
def get_demo_game():
    return find_demo_game(get_model())


@st.cache_data
def get_labels():
    return game_labels(TEST_SEASON)


@st.cache_data
def get_curve(game_id: str):
    return game_curve(get_model(), game_id), game_teams(game_id)


model = get_model()
demo_id, demo_swing = get_demo_game()
labels = get_labels()
_, test_ids = sampled_split()

st.title("🏀 NBA Win Probability — Game Replay")
st.caption(
    "Phase 3 win-probability model. The default game is the test game with the "
    "biggest 4th-quarter swing."
)

options = [demo_id] + [g for g in test_ids if g != demo_id]


def label_for(g: str) -> str:
    base = labels.get(g, g)
    return f"{base}   ⭐ biggest Q4 swing" if g == demo_id else base


game_id = st.selectbox("Game", options=options, format_func=label_for)
curve, (home, away) = get_curve(game_id)
n = len(curve)

left, right = st.columns([3, 2])
with right:
    idx = st.slider("Replay to event", 1, n, n)
    play = st.button("▶ Play from start")

chart = left.empty()
metrics = left.empty()
scorer = left.empty()
feed = right.empty()


def render(k: int):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    plot_winprob(ax, curve, k, home, away)
    chart.pyplot(fig)
    plt.close(fig)

    row = curve.iloc[k - 1]
    with metrics.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{home} (home)", int(row["score_home"]))
        c2.metric(f"{away} (away)", int(row["score_away"]))
        c3.metric("Period", "OT" if row["period"] > 4 else f"Q{int(row['period'])}")
        c4.metric(f"P({home} win)", f"{row['win_prob']:.0%}")

    ls = leading_scorers(curve, k, home, away)
    def top(tri):
        v = ls[tri]
        return f"{v[0]} {v[1]}" if v else "—"
    scorer.markdown(
        f"**Leading scorers** — {home}: {top(home)} · {away}: {top(away)}"
    )

    with feed.container():
        st.markdown("**Play-by-play**")
        seg = curve.iloc[max(0, k - 8):k]
        for _, e in seg.iloc[::-1].iterrows():
            line = f"`{fmt_clock(e['period'], e['secs_left_period'])}`  {e['description']}"
            st.markdown(f"🟢 **{line}**" if e["points"] > 0 else line)


if play:
    step = max(1, n // 200)
    for k in range(1, n + 1, step):
        render(k)
        time.sleep(0.03)
    render(n)
else:
    render(idx)
