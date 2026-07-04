"""Phase 3 - Streamlit win-probability replay dashboard.

Replays a historical game with a live-updating win-probability curve, using the
Phase 3 logistic baseline. Needs the cached play-by-play in data/ (local only).

Run:  streamlit run src/phase3/dashboard.py
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
    find_demo_game, game_curve, game_teams, sampled_split, train_model,
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
def get_curve(game_id: str):
    return game_curve(get_model(), game_id), game_teams(game_id)


model = get_model()
demo_id, demo_swing = get_demo_game()
_, test_ids = sampled_split()

st.title("🏀 NBA Win Probability — Game Replay")
st.caption(
    "Phase 3 logistic baseline. The demo game is the test game with the biggest "
    "4th-quarter win-probability swing."
)

options = [demo_id] + [g for g in test_ids if g != demo_id]
game_id = st.selectbox(
    "Game",
    options=options,
    format_func=lambda g: (f"{g}  — demo: biggest Q4 swing ({demo_swing:.0%})"
                           if g == demo_id else g),
)

curve, (home, away) = get_curve(game_id)
n = len(curve)

col_a, col_b = st.columns([3, 1])
with col_b:
    idx = st.slider("Replay to event", 1, n, n)
    play = st.button("▶ Play from start")

chart = col_a.empty()
score_row = st.empty()


def render(k: int):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    plot_winprob(ax, curve, k, home, away)
    chart.pyplot(fig)
    plt.close(fig)
    row = curve.iloc[k - 1]
    with score_row.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{home} (home)", int(row["score_home"]))
        m2.metric(f"{away} (away)", int(row["score_away"]))
        m3.metric("Period", "OT" if row["period"] > 4 else f"Q{int(row['period'])}")
        m4.metric(f"P({home} win)", f"{row['win_prob']:.0%}")


if play:
    step = max(1, n // 200)
    for k in range(1, n + 1, step):
        render(k)
        time.sleep(0.03)
    render(n)
else:
    render(idx)
