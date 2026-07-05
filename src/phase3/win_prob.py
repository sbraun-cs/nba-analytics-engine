"""Phase 3 - Win probability model (PARSER STAGE ONLY).

This file currently contains just the play-by-play parser and the train/test
split design. No model is trained yet: per CLAUDE.md, the parser is built and
validated on a single game, and the split is proven leak-free, before any
season-wide pull or modeling. The PyTorch/logistic model comes after review.

Run directly (parses ONE cached game + demonstrates the split check):
    python -m src.phase3.win_prob
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.ingest.nba_data import league_game_log, play_by_play, recent_seasons

# A single, ordinary game used to build and sanity-check the parser.
SAMPLE_GAME_ID = "0022300001"  # 2023-24: IND (home) 121, CLE (away) 116

PERIOD_SECONDS = 720   # a regulation quarter is 12 minutes
OT_SECONDS = 300       # an overtime period is 5 minutes

# One-sentence description of every column the parser emits (kept next to the
# code so the schema is self-documenting).
COLUMN_DOC = {
    "game_id":          "NBA game identifier (10-char zero-padded string); constant within a game.",
    "action_number":    "The event's sequence number within the game (ordering key).",
    "period":           "Quarter number; 1-4 for regulation, 5+ for overtime periods.",
    "is_ot":            "1 if this event is in an overtime period (period >= 5), else 0.",
    "secs_left_period": "Seconds remaining on the game clock in the current period (parsed from the ISO 'PT..M..S' clock).",
    "secs_left_game":   "Seconds remaining to the end of regulation (period<=4); during overtime, seconds left in the current OT period (use with is_ot).",
    "score_home":       "Home team's running score after this event (forward-filled from scoring events; starts at 0).",
    "score_away":       "Away team's running score after this event (forward-filled from scoring events; starts at 0).",
    "score_margin":     "score_home minus score_away (positive = home leading) - the core state feature.",
    "action_type":      "Human-readable event category from the API (e.g. 'Made Shot', 'Rebound', 'Turnover', 'Foul', 'Timeout').",
    "home_event":       "1 if the home team performed the action, 0 if the away team - a possession-side proxy. Neutral non-play events are removed (see NEUTRAL policy).",
    "home_win":         "Label: 1 if the home team won the game (from the final score); constant within a game.",
    # Display-only (not model features), used by the replay dashboard:
    "description":      "The API's text description of the event (e.g. 'Tatum 26' 3PT Jump Shot').",
    "player_name":      "Name of the player involved in the event ('' if none).",
    "team_tricode":     "Tricode of the acting team (e.g. 'BOS').",
    "points":           "Points scored on this event (0 for non-scoring events).",
}

# NEUTRAL-EVENT POLICY (explicit decision, not an accident):
# Period start/end markers have no acting team, so home_event is undefined for
# them. We DROP those rows rather than forward-filling the proxy, because they
# are not plays and the adjacent real events already capture the same
# (score_margin, seconds-remaining) game state -- so nothing is lost, and every
# emitted row is a genuine play by a known team. Set drop_neutral=False to keep
# them (they will have home_event = NaN).

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def parse_clock(clock: str) -> float:
    """Turn an ISO-8601 duration like 'PT11M41.00S' into seconds remaining (701.0)."""
    m = _CLOCK_RE.match(str(clock))
    if not m:
        return np.nan
    return int(m.group(1)) * 60 + float(m.group(2))


def parse_game(game_id: str, drop_neutral: bool = True) -> pd.DataFrame:
    """Parse one game's raw play-by-play into per-event win-probability features.

    drop_neutral (default True): remove period start/end markers, which have no
    acting team. See the NEUTRAL-EVENT POLICY note above for the rationale.
    """
    raw = play_by_play(game_id)

    out = pd.DataFrame()
    out["game_id"] = raw["gameId"].astype(str).str.zfill(10)
    out["action_number"] = raw["actionNumber"].astype(int)
    out["period"] = raw["period"].astype(int)
    out["is_ot"] = (out["period"] >= 5).astype(int)

    out["secs_left_period"] = raw["clock"].map(parse_clock)
    reg_left = (4 - out["period"]).clip(lower=0) * PERIOD_SECONDS + out["secs_left_period"]
    out["secs_left_game"] = np.where(out["period"] <= 4, reg_left, out["secs_left_period"])

    # scoreHome/scoreAway are only populated on scoring events -> forward-fill.
    home = pd.to_numeric(raw["scoreHome"], errors="coerce").ffill().fillna(0)
    away = pd.to_numeric(raw["scoreAway"], errors="coerce").ffill().fillna(0)
    out["score_home"] = home.astype(int)
    out["score_away"] = away.astype(int)
    out["score_margin"] = out["score_home"] - out["score_away"]

    out["action_type"] = raw["actionType"].fillna("").astype(str)
    out["home_event"] = raw["location"].map({"h": 1, "v": 0})  # NaN for neutral

    # Display-only fields (not model features): used by the replay dashboard's
    # play-by-play feed and running leading-scorer line.
    out["description"] = raw["description"].fillna("").astype(str)
    out["player_name"] = raw["playerName"].fillna("").astype(str)
    out["team_tricode"] = raw["teamTricode"].fillna("").astype(str)
    # Points scored on this event = increase in the combined score.
    total = out["score_home"] + out["score_away"]
    out["points"] = total.diff().fillna(total.iloc[0]).clip(lower=0).astype(int)

    # Label from the final score of the game (computed BEFORE dropping any rows).
    final_home = int(out["score_home"].iloc[-1])
    final_away = int(out["score_away"].iloc[-1])
    out["home_win"] = int(final_home > final_away)

    if drop_neutral:
        out = out[out["home_event"].notna()].reset_index(drop=True)
        out["home_event"] = out["home_event"].astype(int)
    return out


# --- train/test split design (game- and season-based, proven leak-free) ------
def game_ids_by_season(seasons: list[str]) -> dict[str, set[str]]:
    """Map each season -> the set of its GAME_IDs, read from cached game logs.

    Uses the already-cached LeagueGameLog data (no new API calls) purely to
    demonstrate the split; the win-prob model itself will parse PBP per game.
    """
    out: dict[str, set[str]] = {}
    for s in seasons:
        log = league_game_log(s)
        out[s] = set(log["GAME_ID"].astype(str).str.zfill(10).unique())
    return out


def season_game_split(ids_by_season: dict[str, set[str]], n_test_seasons: int):
    """Whole games from the latest season(s) -> test; all earlier games -> train.

    Splitting by *whole games grouped into seasons* guarantees two things at
    once: the split is time-based (CLAUDE.md rule 2) AND no single game's events
    are ever split across train and test (which would leak the outcome).
    """
    seasons = sorted(ids_by_season)
    test_seasons = seasons[-n_test_seasons:]
    train_ids: set[str] = set()
    test_ids: set[str] = set()
    for s, ids in ids_by_season.items():
        (test_ids if s in test_seasons else train_ids).update(ids)
    return train_ids, test_ids, test_seasons


def assert_no_game_overlap(train_ids: set[str], test_ids: set[str]) -> int:
    """Fail loudly if any GAME_ID appears in both splits. Returns overlap count (0)."""
    overlap = train_ids & test_ids
    assert not overlap, (
        f"Leakage: {len(overlap)} GAME_IDs in BOTH train and test, "
        f"e.g. {sorted(overlap)[:5]}"
    )
    return len(overlap)


def main():
    # (1) Parse the single sample game and show the schema + a sample of rows.
    game = parse_game(SAMPLE_GAME_ID)
    print(f"Parsed game {SAMPLE_GAME_ID}: {len(game)} events, "
          f"final {game['score_home'].iloc[-1]}-{game['score_away'].iloc[-1]}, "
          f"home_win={game['home_win'].iloc[0]}\n")

    print("Columns:")
    for col, doc in COLUMN_DOC.items():
        print(f"  {col:<17} {doc}")

    show = ["period", "is_ot", "secs_left_period", "secs_left_game", "score_home",
            "score_away", "score_margin", "action_type", "home_event", "home_win"]
    print("\nOpening tip-off:")
    print(game[show].head(6).to_string(index=False))
    print("\nCrunch time (last 6 events):")
    print(game[show].tail(6).to_string(index=False))

    # (2) Demonstrate the game/season split + prove zero GAME_ID overlap, using
    #     the already-cached game logs (no new pulls, no model training).
    print("\n--- Split design check (using cached game logs, 10 seasons) ---")
    ids = game_ids_by_season(recent_seasons(10, 2024))
    train_ids, test_ids, test_seasons = season_game_split(ids, n_test_seasons=1)
    n_overlap = assert_no_game_overlap(train_ids, test_ids)
    print(f"Train games: {len(train_ids):,} | Test games: {len(test_ids):,} "
          f"| Test season(s): {test_seasons}")
    print(f"GAME_IDs in both train and test: {n_overlap}  (assertion passed)")


if __name__ == "__main__":
    main()
