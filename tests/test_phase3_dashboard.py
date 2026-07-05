"""Regression tests for the Phase 3 dashboard curve schema.

Guards the bug where a stale Streamlit cache served a curve built by older code
that lacked the display columns, causing `KeyError: 'points'` in leading_scorers
while scrubbing. Runs standalone (`python tests/test_phase3_dashboard.py`) and is
also discoverable by pytest. Needs the cached play-by-play in data/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.phase3.baseline import (
    REQUIRED_CURVE_COLUMNS, assert_curve_columns, game_curve, game_teams,
    leading_scorers, sampled_split, train_model,
)


def _expect_keyerror(fn):
    try:
        fn()
    except KeyError:
        return True
    raise AssertionError("expected KeyError, none raised")


def test_stale_curve_missing_display_columns_is_caught():
    """A curve without the display columns must fail fast, not silently."""
    stale = pd.DataFrame({"game_id": ["x"], "period": [1]})  # pre-schema curve
    _expect_keyerror(lambda: assert_curve_columns(stale))
    # This is the exact original failure mode (KeyError 'points' at line ~193).
    _expect_keyerror(lambda: leading_scorers(stale, 1, "AAA", "BBB"))


def test_game_curve_has_display_columns_on_non_demo_game():
    """The path the bug was hit on: a NON-demo game, scrubbed to several points."""
    train_ids, test_ids = sampled_split()
    model = train_model(train_ids=train_ids[:20])  # small subset = fast
    non_demo = test_ids[3]

    curve = game_curve(model, non_demo)
    missing = [c for c in REQUIRED_CURVE_COLUMNS if c not in curve.columns]
    assert not missing, f"curve missing {missing}"

    home, away = game_teams(non_demo)
    for upto in (1, len(curve) // 2, len(curve)):   # opening, mid, final
        ls = leading_scorers(curve, upto, home, away)
        assert set(ls) == {home, away}


if __name__ == "__main__":
    test_stale_curve_missing_display_columns_is_caught()
    print("PASS test_stale_curve_missing_display_columns_is_caught")
    test_game_curve_has_display_columns_on_non_demo_game()
    print("PASS test_game_curve_has_display_columns_on_non_demo_game")
    print("All Phase 3 dashboard regression tests passed.")
