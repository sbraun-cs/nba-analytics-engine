# NBA Analytics Engine

A three-phase NBA analytics portfolio project. One shared data pipeline feeds three
models of escalating difficulty:

| Phase | Model | Status |
|-------|-------|--------|
| 1 | Game outcome predictor (logistic regression) | ✅ Complete |
| 2 | Shot quality model — xFG% (XGBoost) | ✅ Complete |
| 3 | Live win-probability model + Streamlit dashboard (PyTorch) | ⬜ Planned |

All data is pulled from the public `nba_api`, cached locally under `data/` (never
committed), and every model is evaluated against an honest naive baseline with a
**time-based** train/test split — no random shuffling, no tuning on the test set.

## Repo layout

```
src/ingest/     shared nba_api access + caching (used by every phase)
src/phase1/     game predictor
notebooks/      one exploration notebook per phase
data/           cached API pulls (gitignored)
```

## Setup

```bash
pip install nba_api pandas scikit-learn xgboost seaborn matplotlib jupyter
```

## Phase 1 — Game predictor

Predicts whether the **home team wins** from each team's recent form and rest,
using 10 seasons of team game logs (2015-16 → 2024-25).

**Run it:**

```bash
python -m src.phase1.game_predictor      # fetches (once) + trains + reports
jupyter notebook notebooks/phase1.ipynb  # plots
```

### Features (all leak-free)

Every rolling feature is computed per (team, season) and **shifted by one game**,
so a game is never part of its own inputs. A runtime leakage guard asserts that the
first games of each team-season have no rolling features. Features are expressed as
home-minus-away differences:

- `d_pts`, `d_pts_allowed` — rolling points scored / allowed (last 10 games)
- `d_rest` — rest-day difference
- *(stretch)* `d_off_rating`, `d_def_rating` — rolling offensive/defensive rating
- *(stretch)* `home_b2b`, `away_b2b` — back-to-back flags

> Rolling win% was tested and dropped: scoring margin (`d_pts`/`d_pts_allowed`)
> subsumes it, and removing it left accuracy unchanged (actually a touch higher).

### Results

Trained on 2015-16 → 2022-23 (8,889 games), tested on the held-out 2023-24 and
2024-25 seasons (2,297 games).

| Model | Accuracy | Log loss |
|-------|----------|----------|
| **Naive baseline** (always pick home) | 0.545 | 0.689 |
| Base (form + rest) | 0.648 | 0.626 |
| Extended (+ ratings + back-to-back) | **0.659** | **0.623** |

The model beats the baseline on both accuracy and log loss. Accuracy sits in the
mid-60s % — in line with what public data supports for NBA game prediction, and
comfortably below the ~70% level that would signal data leakage.

**What matters most** (standardized coefficients): rolling scoring margin
(`d_pts`, `d_pts_allowed`) dominates; rest and back-to-back effects are real but
small. Margin already encodes team strength, which is why rolling win% was
redundant and dropped.

**Rest matters, monotonically.** Home win rate by rest-day advantage:

| Away more rested (≤ -2) | -1 | Even | +1 | Home more rested (≥ +2) |
|:---:|:---:|:---:|:---:|:---:|
| 52.7% | 53.9% | 55.8% | 59.8% | 62.6% |

### Lessons learned

- A single `LeagueGameLog` call per season is far more polite (and faster) than
  pulling each team separately — 10 API calls for 10 seasons.
- Rolling scoring margin carries most of the signal; adding win% on top is nearly
  redundant. Extra features (ratings, back-to-back) gave a small, honest lift.
- Getting the `shift(1)`-then-`rolling` order right — and guarding it with an
  assertion — is the whole ballgame for avoiding leakage.

## Phase 2 — Shot quality model (xFG%)

Predicts the probability a field-goal attempt goes in, from **512,000 shots**
across 5 seasons (2020-21 → 2024-25) of league-wide `shotchartdetail` data.

**Run it:**

```bash
python -m src.phase2.shot_quality       # fetches (once) + trains + reports
jupyter notebook notebooks/phase2.ipynb  # shot chart + calibration
```

### Data limitation (important)

The public `shotchartdetail` endpoint does **not** expose **defender distance,
shot-clock, or score margin** for these seasons — the three signals that would
most improve a shot-difficulty model. Rather than fake them, the model uses only
what the endpoint actually provides:

- `shot_distance`, `loc_x`, `loc_y`, and a derived `shot_angle`
- `is_three`, `period`, `time_remaining_sec` (game clock)
- one-hot `SHOT_ZONE_BASIC` and the 15 most common `ACTION_TYPE`s (layup, jumper,
  dunk, …)

### Results

XGBoost, trained on 2020-21 → 2023-24 (409,600 shots), tested on the held-out
2024-25 season (102,400 shots).

| Model | Log loss | ROC-AUC |
|-------|----------|---------|
| **Naive baseline** (league-average 46.6%) | 0.691 | 0.500 |
| XGBoost xFG% | **0.640** | **0.657** |

A meaningful but modest lift, and well calibrated (predicted probabilities track
observed make rates closely). At-rim shots average a predicted ~0.66 make
probability versus ~0.36 for threes — the model recovers the shape of shot value
from geometry alone. The mid-0.60s AUC is the honest ceiling for a public-data
model without defender or shot-clock context; a much higher number would be a
red flag.

### Lessons learned

- One `ShotChartDetail` call with `team_id=0, player_id=0` pulls an entire
  season league-wide (~100k shots) — 5 calls for 5 seasons.
- Shot-zone (especially the restricted area) and distance dominate importance;
  the missing defender/shot-clock fields are the real accuracy ceiling, and
  naming that limitation honestly matters more than squeezing out AUC.
