"""Shared NBA API access for every phase of the project.

All calls to nba_api go through here so that phases 1-3 reuse the same
caching + rate-limiting logic. Rules (see CLAUDE.md):
  - sleep >= 1.5s between live API calls
  - cache every response to data/ and read from cache when it exists
  - never re-fetch data that is already cached
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

# data/ lives at the project root: .../nba-analytics-engine/data
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)

# Polite delay between live API calls (seconds). nba_api is rate-limited.
API_SLEEP_SECONDS = 1.6


def season_str(start_year: int) -> str:
    """Return the nba_api season string, e.g. 2015 -> '2015-16'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def recent_seasons(n: int, last_start_year: int) -> list[str]:
    """The n most recent seasons ending with `last_start_year`.

    e.g. recent_seasons(3, 2024) -> ['2022-23', '2023-24', '2024-25'].
    """
    return [season_str(y) for y in range(last_start_year - n + 1, last_start_year + 1)]


def _cache_path(name: str) -> Path:
    return DATA_DIR / name


def league_game_log(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """One row per team per game for a full season (both teams -> two rows/game).

    Uses the LeagueGameLog endpoint: a single API call covers an entire season,
    which is far more polite than pulling each team separately. Cached as CSV.
    """
    safe_type = season_type.replace(" ", "_").lower()
    cache = _cache_path(f"league_game_log_{season}_{safe_type}.csv")

    if cache.exists():
        return pd.read_csv(cache)

    # Import here so importing this module doesn't require nba_api unless we fetch.
    from nba_api.stats.endpoints import leaguegamelog

    print(f"[nba_data] fetching LeagueGameLog {season} ({season_type})...")
    resp = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
    )
    df = resp.get_data_frames()[0]
    time.sleep(API_SLEEP_SECONDS)  # be polite to the API

    df.to_csv(cache, index=False)
    return df


def team_game_logs(seasons: list[str], season_type: str = "Regular Season") -> pd.DataFrame:
    """Concatenate LeagueGameLog for several seasons into one long table."""
    frames = [league_game_log(s, season_type) for s in seasons]
    out = pd.concat(frames, ignore_index=True)
    return out


def shot_chart(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Every field-goal attempt league-wide for a season (~100k rows), cached.

    Uses ShotChartDetail with team_id=0/player_id=0 to pull all shooters at once.
    Note: this endpoint has NO defender-distance, shot-clock, or score-margin
    fields -- see the Phase 2 README for how that limits the model.
    """
    safe_type = season_type.replace(" ", "_").lower()
    cache = _cache_path(f"shot_chart_{season}_{safe_type}.csv")

    if cache.exists():
        return pd.read_csv(cache, dtype={"GAME_ID": str})

    from nba_api.stats.endpoints import shotchartdetail

    print(f"[nba_data] fetching ShotChartDetail {season} ({season_type})...")
    resp = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,
        season_nullable=season,
        season_type_all_star=season_type,
        context_measure_simple="FGA",  # all field-goal attempts
    )
    df = resp.get_data_frames()[0]
    time.sleep(API_SLEEP_SECONDS)  # be polite to the API

    df.to_csv(cache, index=False)
    return df


def shot_charts(seasons: list[str], season_type: str = "Regular Season") -> pd.DataFrame:
    """Concatenate ShotChartDetail for several seasons into one long table."""
    frames = []
    for s in seasons:
        df = shot_chart(s, season_type)
        df["SEASON"] = s  # tag each shot with its season for time-based splits
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
