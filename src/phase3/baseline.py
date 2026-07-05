"""Phase 3 - logistic win-probability BASELINE (a simple model before the net).

Samples a modest set of games (per the phase gate), pulls their play-by-play
once and caches it, parses each with the Phase 3 parser, and fits a logistic
regression on the clean core features. Evaluated on a held-out later season with
log loss, ROC-AUC, a naive baseline, and calibration by game-time bucket.

Two entry points:
    python -m src.phase3.baseline pull     # fetch + cache the sampled games (slow, rate-limited)
    python -m src.phase3.baseline train    # parse from cache + fit + evaluate (fast)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest.nba_data import league_game_log, play_by_play
from src.phase3.win_prob import assert_no_game_overlap, parse_game

# --- sampling configuration (modest first pull) ------------------------------
TRAIN_SEASONS = ["2022-23", "2023-24"]
TEST_SEASON = "2024-25"
N_TRAIN_PER_SEASON = 150
N_TEST = 150
SEED = 0

# margin_urgency: a lead is worth more with less time left. score_margin divided
# by (minutes remaining + 1) so it grows from ~0 early to the raw margin at 0:00.
FEATURES = ["score_margin", "secs_left_game", "margin_urgency", "is_ot", "home_event"]


def sample_ids(season: str, n: int, seed: int) -> list[str]:
    """Deterministically sample n GAME_IDs from a season's cached game log."""
    log = league_game_log(season)
    ids = pd.Series(sorted(log["GAME_ID"].astype(str).str.zfill(10).unique()))
    return list(ids.sample(n=min(n, len(ids)), random_state=seed))


def sampled_split() -> tuple[list[str], list[str]]:
    """Train GAME_IDs (earlier seasons) and test GAME_IDs (latest season)."""
    train_ids: list[str] = []
    for i, s in enumerate(TRAIN_SEASONS):
        train_ids += sample_ids(s, N_TRAIN_PER_SEASON, SEED + i)
    test_ids = sample_ids(TEST_SEASON, N_TEST, SEED + 99)
    assert_no_game_overlap(set(train_ids), set(test_ids))  # never split a game
    return train_ids, test_ids


def pull(ids: list[str]) -> None:
    """Fetch + cache play-by-play for every game id (the slow, rate-limited step)."""
    failed = []
    for i, g in enumerate(ids, 1):
        try:
            play_by_play(g)
        except Exception as e:  # keep going; report at the end
            failed.append((g, f"{type(e).__name__}: {e}"))
        if i % 25 == 0 or i == len(ids):
            print(f"  pulled {i}/{len(ids)} (failures so far: {len(failed)})", flush=True)
    if failed:
        print(f"WARNING: {len(failed)} games failed to fetch: {failed[:5]}")


def build_dataset(ids: list[str]) -> pd.DataFrame:
    """Parse every cached game into one per-event feature table."""
    frames = []
    for g in ids:
        try:
            frames.append(parse_game(g))
        except Exception as e:
            print(f"  skip {g}: {type(e).__name__}: {e}")
    data = pd.concat(frames, ignore_index=True)
    minutes_left = data["secs_left_game"] / 60.0
    data["margin_urgency"] = data["score_margin"] / (minutes_left + 1.0)
    return data.dropna(subset=FEATURES).reset_index(drop=True)


def train_model(train_ids: list[str] | None = None):
    """Fit the logistic win-prob model (reused by the dashboard and GIF)."""
    if train_ids is None:
        train_ids, _ = sampled_split()
    train = build_dataset(train_ids)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[FEATURES], train["home_win"])
    return model


def _elapsed_seconds(period: int, secs_left_period: float) -> float:
    """Monotonic seconds since tip-off (handles OT), for a left-to-right time axis."""
    if period <= 4:
        return (period - 1) * 720 + (720 - secs_left_period)
    return 2880 + (period - 5) * 300 + (300 - secs_left_period)


def game_curve(model, game_id: str) -> pd.DataFrame:
    """Per-event win-probability curve for one game (predicted P(home win))."""
    df = build_dataset([game_id]).copy()
    df["win_prob"] = model.predict_proba(df[FEATURES])[:, 1]
    df["secs_elapsed"] = [
        _elapsed_seconds(p, s) for p, s in zip(df["period"], df["secs_left_period"])
    ]
    df["minutes_elapsed"] = df["secs_elapsed"] / 60.0
    return df.sort_values("secs_elapsed").reset_index(drop=True)


def q4_swing(model, game_id: str) -> float:
    """Range (max - min) of the win-prob curve during the 4th quarter."""
    df = game_curve(model, game_id)
    q4 = df[df["period"] == 4]
    if len(q4) < 2:
        return 0.0
    return float(q4["win_prob"].max() - q4["win_prob"].min())


def find_demo_game(model=None, test_ids: list[str] | None = None):
    """Return (game_id, swing) for the test game with the biggest 4th-quarter swing."""
    if model is None:
        model = train_model()
    if test_ids is None:
        _, test_ids = sampled_split()
    best_id, best_swing = None, -1.0
    for g in test_ids:
        try:
            swing = q4_swing(model, g)
        except Exception:
            continue
        if swing > best_swing:
            best_id, best_swing = g, swing
    return best_id, best_swing


def game_teams(game_id: str) -> tuple[str, str]:
    """(home_tricode, away_tricode) for a game, read from its cached play-by-play."""
    raw = play_by_play(game_id)
    home = raw.loc[raw["location"] == "h", "teamTricode"].dropna().iloc[0]
    away = raw.loc[raw["location"] == "v", "teamTricode"].dropna().iloc[0]
    return home, away


def game_labels(season: str) -> dict[str, str]:
    """game_id -> 'DATE — AWAY @ HOME (away-home)' from the cached game log."""
    log = league_game_log(season).copy()
    log["gid"] = log["GAME_ID"].astype(str).str.zfill(10)
    labels: dict[str, str] = {}
    for gid, grp in log.groupby("gid"):
        home = grp[grp["MATCHUP"].str.contains("vs.", regex=False)]
        away = grp[grp["MATCHUP"].str.contains("@", regex=False)]
        if home.empty or away.empty:
            continue
        h, a = home.iloc[0], away.iloc[0]
        date = str(h["GAME_DATE"])[:10]
        labels[gid] = (f"{date} — {a['TEAM_ABBREVIATION']} @ {h['TEAM_ABBREVIATION']} "
                       f"({int(a['PTS'])}-{int(h['PTS'])})")
    return labels


def leading_scorers(curve: pd.DataFrame, upto: int, home: str, away: str) -> dict:
    """Top scorer (name, points) for each team using scoring events up to `upto`."""
    seg = curve.iloc[: max(1, upto)]
    made = seg[seg["points"] > 0]
    out = {}
    for tri in (home, away):
        t = made[made["team_tricode"] == tri]
        if t.empty:
            out[tri] = None
        else:
            totals = t.groupby("player_name")["points"].sum()
            out[tri] = (totals.idxmax(), int(totals.max()))
    return out


def fmt_clock(period: int, secs_left_period: float) -> str:
    """Format the game clock, e.g. (4, 41.0) -> 'Q4 0:41', (5, 120) -> 'OT 2:00'."""
    minutes = int(secs_left_period // 60)
    seconds = int(round(secs_left_period % 60))
    quarter = "OT" if period > 4 else f"Q{int(period)}"
    return f"{quarter} {minutes}:{seconds:02d}"


def calibration_by_period(test: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """Mean predicted vs. actual home-win rate, bucketed by period."""
    t = test.copy()
    t["pred"] = prob
    label = t["period"].apply(lambda p: f"Q{p}" if p <= 4 else "OT")
    g = t.groupby(label).agg(events=("pred", "size"),
                             mean_pred=("pred", "mean"),
                             actual=("home_win", "mean"))
    return g


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> None:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[FEATURES], train["home_win"])
    prob = model.predict_proba(test[FEATURES])[:, 1]

    ll = log_loss(test["home_win"], prob)
    auc = roc_auc_score(test["home_win"], prob)

    # Naive baseline: predict the training base rate for every event.
    base = train["home_win"].mean()
    base_ll = log_loss(test["home_win"], np.full(len(test), base))

    print("\n=== Logistic win-probability baseline ===")
    print(f"  train events : {len(train):,}  (games: {train['game_id'].nunique()})")
    print(f"  test events  : {len(test):,}  (games: {test['game_id'].nunique()})")
    print(f"  base rate    : {base:.4f}")
    print(f"  log loss     : {ll:.4f}   (naive baseline {base_ll:.4f})")
    print(f"  ROC-AUC      : {auc:.4f}")

    coefs = pd.Series(model.named_steps["logisticregression"].coef_[0], index=FEATURES)
    print("  coefficients :")
    for name, val in coefs.sort_values(key=abs, ascending=False).items():
        print(f"    {name:<15} {val:+.4f}")

    print("\n  calibration by period (mean predicted vs actual home-win rate):")
    print(calibration_by_period(test, prob).round(4).to_string())


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    train_ids, test_ids = sampled_split()

    if cmd == "pull":
        all_ids = train_ids + test_ids
        print(f"Pulling play-by-play for {len(all_ids)} games "
              f"({len(train_ids)} train + {len(test_ids)} test)...", flush=True)
        pull(all_ids)
        print("Pull complete.")
        return

    if cmd == "demo":
        model = train_model(train_ids)
        game_id, swing = find_demo_game(model, test_ids)
        home, away = game_teams(game_id)
        curve = game_curve(model, game_id)
        print(f"Biggest 4th-quarter swing: game {game_id} ({away} @ {home}), "
              f"Q4 swing = {swing:.3f}")
        print(f"Final: {home} {curve['score_home'].iloc[-1]} - "
              f"{curve['score_away'].iloc[-1]} {away}")
        return

    print(f"Building dataset from {len(train_ids)} train + {len(test_ids)} test games...")
    train = build_dataset(train_ids)
    test = build_dataset(test_ids)
    evaluate(train, test)


if __name__ == "__main__":
    main()
