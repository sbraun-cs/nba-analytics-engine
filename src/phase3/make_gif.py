"""Render the demo-game win-probability replay to a GIF for the README.

    python -m src.phase3.make_gif

Picks the test game with the biggest 4th-quarter swing, then animates the
win-probability curve revealing over game time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.phase3.baseline import find_demo_game, game_curve, game_teams, train_model
from src.phase3.viz import plot_winprob

OUT = Path("docs/phase3_winprob_replay.gif")


def main():
    model = train_model()
    game_id, swing = find_demo_game(model)
    home, away = game_teams(game_id)
    curve = game_curve(model, game_id)
    n = len(curve)
    print(f"Demo game {game_id}: {away} @ {home}, Q4 swing {swing:.3f}, {n} events")

    # Reveal the curve in ~100 frames, then hold on the final frame.
    step = max(1, n // 100)
    frames = list(range(1, n + 1, step)) + [n] * 12

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=90)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.13)

    def update(k):
        plot_winprob(ax, curve, k, home, away)

    anim = FuncAnimation(fig, update, frames=frames, interval=80)
    OUT.parent.mkdir(exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=14))
    plt.close(fig)
    size_mb = OUT.stat().st_size / 1e6
    print(f"Saved {OUT} ({len(frames)} frames, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
