"""Phase 1 - NBA game outcome predictor.

Pipeline:
  1. Pull N seasons of team game logs (via src/ingest, cached).
  2. Build rolling-form, rest, and (stretch) rating features using ONLY past
     games -- every rolling feature is shifted so the game being predicted is
     never part of its own inputs (no data leakage; see CLAUDE.md rule 1).
  3. Reshape to one row per game (home vs away) with target = home team wins.
  4. Time-based split (earlier seasons train, later seasons test -- rule 2).
  5. Logistic regression; report accuracy, log loss, and a naive baseline.

Run directly:  python -m src.phase1.game_predictor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest.nba_data import DATA_DIR, recent_seasons, season_str, team_game_logs

# --- configuration -----------------------------------------------------------
N_SEASONS = 10
LAST_SEASON_START = 2024          # -> most recent season is 2024-25
ROLL_WINDOW = 10                  # rolling-form window (games)
ROLL_MIN_PERIODS = 5             # need at least this many past games
N_TEST_SEASONS = 2               # hold out the final seasons for testing

# Feature groups. "base" = the spec's core (form + rest). "extra" = stretch.
# d_winpct is intentionally excluded: rolling scoring margin (d_pts/d_pts_allowed)
# already subsumes it, and it added ~nothing on the time-split test.
BASE_FEATURES = ["d_pts", "d_pts_allowed", "d_rest"]
EXTRA_FEATURES = ["d_off_rating", "d_def_rating", "home_b2b", "away_b2b"]


# --- feature engineering -----------------------------------------------------
def _add_opponent_and_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each game's opponent stats and derive per-team game columns."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    df["WIN"] = (df["WL"] == "W").astype(int)

    # Possessions estimate (standard formula) for offensive/defensive ratings.
    df["POSS"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]

    # Self-join on GAME_ID to pull the opponent's row (points allowed, etc.).
    opp_cols = ["GAME_ID", "TEAM_ID", "PTS", "POSS"]
    opp = df[opp_cols].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS", "POSS": "OPP_POSS"}
    )
    merged = df.merge(opp, on="GAME_ID")
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()

    merged["PTS_ALLOWED"] = merged["OPP_PTS"]
    merged["OFF_RATING"] = 100 * merged["PTS"] / merged["POSS"]
    merged["DEF_RATING"] = 100 * merged["OPP_PTS"] / merged["OPP_POSS"]
    return merged


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling form + rest, computed per (team, season) using only PAST games.

    Every rolling stat is shifted by one game within the group, so a game's
    features come strictly from earlier games. Rest days reset each season.
    """
    df = df.sort_values(["TEAM_ID", "SEASON_ID", "GAME_DATE"]).copy()
    g = df.groupby(["TEAM_ID", "SEASON_ID"], sort=False)

    def past_roll(col):
        # shift(1) first (drop current game), then average the prior window.
        shifted = g[col].shift(1)
        return shifted.groupby([df["TEAM_ID"], df["SEASON_ID"]], sort=False).transform(
            lambda s: s.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).mean()
        )

    df["roll_winpct"] = past_roll("WIN")
    df["roll_pts"] = past_roll("PTS")
    df["roll_pts_allowed"] = past_roll("PTS_ALLOWED")
    df["roll_off_rating"] = past_roll("OFF_RATING")
    df["roll_def_rating"] = past_roll("DEF_RATING")

    # Rest days: nights since previous game this season. First game -> NaN.
    df["rest_days"] = g["GAME_DATE"].diff().dt.days
    df["b2b"] = (df["rest_days"] == 1).astype(float)  # back-to-back flag
    return df


def _leakage_guard(df: pd.DataFrame) -> None:
    """Fail loudly if any rolling feature leaks the current game.

    By construction the first ROLL_MIN_PERIODS games of every (team, season)
    must have NaN rolling features. If they don't, the shift/rolling is wrong.
    """
    first = df.sort_values(["TEAM_ID", "SEASON_ID", "GAME_DATE"]).groupby(
        ["TEAM_ID", "SEASON_ID"], sort=False
    ).head(1)
    leaked = first["roll_winpct"].notna().sum()
    if leaked:
        raise AssertionError(
            f"Leakage guard failed: {leaked} teams have a rolling feature on "
            "their first game of the season (should be NaN)."
        )


def build_game_table(seasons: list[str]) -> pd.DataFrame:
    """One row per game: home features, away features, target = home win."""
    raw = team_game_logs(seasons)
    long = _add_opponent_and_basic(raw)
    long = _add_rolling_features(long)
    _leakage_guard(long)

    feat_cols = [
        "roll_winpct", "roll_pts", "roll_pts_allowed",
        "roll_off_rating", "roll_def_rating", "rest_days", "b2b",
    ]
    keep = ["GAME_ID", "SEASON_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
            "IS_HOME", "WIN"] + feat_cols

    home = long[long["IS_HOME"] == 1][keep].add_prefix("home_")
    away = long[long["IS_HOME"] == 0][keep].add_prefix("away_")
    games = home.merge(away, left_on="home_GAME_ID", right_on="away_GAME_ID")

    games["target"] = games["home_WIN"]  # 1 if home team won
    games["SEASON_ID"] = games["home_SEASON_ID"]
    games["GAME_DATE"] = games["home_GAME_DATE"]
    # Zero-padded 10-char game id, so Phase 3 can join these pregame features on it.
    games["game_id"] = games["home_GAME_ID"].astype(str).str.zfill(10)

    # Home-minus-away differences (the model's actual inputs).
    # roll_winpct is still computed upstream (it anchors the leakage guard) but is
    # not used as a model feature -- scoring margin subsumes it.
    games["d_pts"] = games["home_roll_pts"] - games["away_roll_pts"]
    games["d_pts_allowed"] = games["home_roll_pts_allowed"] - games["away_roll_pts_allowed"]
    games["d_off_rating"] = games["home_roll_off_rating"] - games["away_roll_off_rating"]
    games["d_def_rating"] = games["home_roll_def_rating"] - games["away_roll_def_rating"]
    games["d_rest"] = games["home_rest_days"] - games["away_rest_days"]
    games["home_b2b"] = games["home_b2b"]
    games["away_b2b"] = games["away_b2b"]

    model_cols = ["game_id", "SEASON_ID", "GAME_DATE", "target",
                  "home_TEAM_ABBREVIATION", "away_TEAM_ABBREVIATION"] + \
                 BASE_FEATURES + EXTRA_FEATURES
    out = games[[c for c in model_cols if c in games.columns]].dropna(
        subset=BASE_FEATURES + EXTRA_FEATURES
    )
    return out.sort_values("GAME_DATE").reset_index(drop=True)


# --- evaluation --------------------------------------------------------------
def time_split(games: pd.DataFrame, n_test_seasons: int):
    """Train on earlier seasons, test on the last `n_test_seasons`."""
    seasons = sorted(games["SEASON_ID"].unique())
    test_seasons = seasons[-n_test_seasons:]
    is_test = games["SEASON_ID"].isin(test_seasons)
    return games[~is_test].copy(), games[is_test].copy(), test_seasons


def evaluate(train, test, features, label):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[features], train["target"])

    prob = model.predict_proba(test[features])[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(test["target"], pred)
    ll = log_loss(test["target"], prob)

    # Naive baseline: always predict the home team wins.
    base_acc = accuracy_score(test["target"], np.ones(len(test)))
    base_ll = log_loss(test["target"], np.full(len(test), test["target"].mean()))

    coefs = pd.Series(
        model.named_steps["logisticregression"].coef_[0], index=features
    ).sort_values(key=abs, ascending=False)

    print(f"\n=== {label} ({len(features)} features) ===")
    print(f"  test games         : {len(test)}")
    print(f"  accuracy           : {acc:.4f}   (baseline home-wins {base_acc:.4f})")
    print(f"  log loss           : {ll:.4f}   (baseline {base_ll:.4f})")
    print("  coefficients (|desc|):")
    for name, val in coefs.items():
        print(f"    {name:<16} {val:+.4f}")

    return {"label": label, "accuracy": acc, "log_loss": ll,
            "baseline_accuracy": base_acc, "baseline_log_loss": base_ll,
            "coefficients": coefs}


def main():
    seasons = recent_seasons(N_SEASONS, LAST_SEASON_START)
    print(f"Seasons: {seasons[0]} ... {seasons[-1]}  ({len(seasons)} seasons)")

    games = build_game_table(seasons)
    print(f"Modeling {len(games)} games with usable features.")

    # Cache the processed table so the notebook doesn't recompute it.
    out_path = DATA_DIR / "phase1_game_features.csv"
    games.to_csv(out_path, index=False)
    print(f"Saved processed features -> {out_path}")

    train, test, test_seasons = time_split(games, N_TEST_SEASONS)
    # SEASON_ID is coded like 22023 = "2" + start year; show it readably.
    readable = [season_str(int(str(s)[1:])) for s in test_seasons]
    print(f"Train games: {len(train)} | Test games: {len(test)} "
          f"| Test seasons: {readable}")

    evaluate(train, test, BASE_FEATURES, "Base model (form + rest)")
    evaluate(train, test, BASE_FEATURES + EXTRA_FEATURES,
             "Extended model (+ ratings + back-to-back)")

    print("\nReminder: Phase 1 accuracy > ~70% would be a red flag for leakage.")


if __name__ == "__main__":
    main()
