# NBA Analytics Engine

A three-phase NBA analytics portfolio project. One shared data pipeline feeds three
models of escalating difficulty:

| Phase | Model | Status |
|-------|-------|--------|
| 1 | Game outcome predictor (logistic regression) | ✅ Complete |
| 2 | Shot quality model — xFG% (XGBoost) | ✅ Complete |
| 3 | Live win-probability model + Streamlit dashboard (PyTorch) | 🔨 In progress (parser) |

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

### A COVID natural experiment (home crowds)

Part of 2019-20 (the Orlando "bubble") and much of 2020-21 were played in empty or
reduced-capacity arenas. If home-court advantage is partly a *crowd* effect, home
win% should dip in exactly those seasons — and it does:

| 2015-16 → 2018-19 | 2019-20 | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **58.6%** (avg) | 55.1% | 54.4% | 54.4% | 58.1% | 54.3% | 54.5% |

Home win% falls ~4.5 points in the empty-arena seasons — direct, if circumstantial,
evidence of the crowd's contribution to home advantage, detected inside our own
pipeline with no extra data. The honest caveat: it never fully rebounds to the
pre-COVID ~59%, so a broader league-wide decline in home-court advantage is also at
work — the crowd effect is real but not the whole story.

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

> The COVID-affected 2020-21 season is kept deliberately. Empty arenas plausibly
> affect *game outcomes* (see the Phase 1 crowd analysis) but there's little reason
> a *shot's* make probability given its location changes with crowd size, so it's
> fair training data here — and dropping ~100k shots would cost more than it buys.

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

## Phase 3 — Win probability model + dashboard (in progress)

Predicts P(home win) at each moment of a game from live game state. Built
incrementally per the brief: the play-by-play **parser is validated on a single
game first**, and the train/test split is proven leak-free, before any
season-wide pull or model training.

- **Data:** `PlayByPlayV3` (the older `playbyplayv2` is deprecated — the NBA API
  now returns empty JSON for it), fetched and cached one game at a time.
- **Core features per event:** `score_margin`, `secs_left_game`, `period`,
  `is_ot`, and `home_event` (a possession-side proxy). A logistic model is the
  baseline before any neural net.
- **Split:** whole games grouped by season (latest season = test). Because a
  whole game is the atomic unit, no game's events are ever split across train and
  test — enforced by an assertion that the train/test `GAME_ID` sets are disjoint.

### Stretch idea (v2, after the dashboard works): reuse Phase 1 as a prior

A pure in-game win-probability model starts *every* game near 50/50, ignoring who
is playing. The plan is to feed **Phase 1's pregame features** (rolling scoring-
margin difference, rest) into Phase 3 as prior features, so a strong team vs. a
weak team opens where it should instead of a coin flip. That turns three separate
projects into **one engine** — the game-level model becomes the prior for the
event-level model — which is the whole point of the repo's name.
