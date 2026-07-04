# CLAUDE.md — nba-analytics-engine

## What this project is
A three-phase NBA analytics portfolio project for a data science student. One shared
data pipeline feeds three models of escalating difficulty. This repo will be shown to
recruiters: code quality, honest evaluation, and a clear README matter as much as accuracy.

The owner is learning. Explain non-obvious changes in one or two sentences when you make
them. Prefer readable code over clever code. Do not add heavy abstractions or frameworks
that obscure what the code does.

## Repo layout
- `data/` — cached API pulls (CSV/parquet). Never committed (gitignore). Never re-fetch
  data that is already cached.
- `src/ingest/` — all NBA API access lives here, shared by every phase.
- `src/phase1/`, `src/phase2/`, `src/phase3/` — one folder per model.
- `notebooks/` — exploration and plots, one notebook per phase.
- `README.md` — updated at the end of every phase with results and lessons.

## Hard rules (do not violate, do not "optimize away")
1. **No data leakage.** All rolling/aggregate features must use only PAST games
   (`.shift(1)` pattern or equivalent). Any feature that could contain information from
   the game being predicted is a bug, even if it improves accuracy.
2. **Time-based splits only.** Train on earlier seasons, test on later ones. Never
   random-shuffle splits. Never tune on the test set.
3. **Polite API usage.** `nba_api` is rate-limited: sleep >= 1.5s between calls, cache
   every response to `data/`, and read from cache when it exists.
4. **Honest metrics.** Always report a naive baseline next to model results (e.g.
   home-team-wins for Phase 1, always-miss for Phase 2). Report log loss / calibration,
   not just accuracy. If test accuracy looks too good (Phase 1 > ~70%, Phase 2 AUC > ~0.75),
   suspect leakage and investigate before celebrating.
5. **Small commits with clear messages**, roughly one per working feature. Never commit
   `data/` contents or credentials.

## Phase 1 — Game predictor (exists, needs verification + polish)
`src/phase1_game_predictor.py` pulls 10 seasons of team game logs, builds rolling-form
and rest-day features, trains logistic regression with a time split.
Tasks:
- Run it end to end. Fix any nba_api column/endpoint changes. Keep the leakage guard
  and time split intact.
- Move API code into `src/ingest/` so later phases reuse it.
- Add 2–3 seaborn plots in `notebooks/phase1.ipynb` (e.g. win% by rest-day advantage,
  calibration curve).
- Report: accuracy, log loss, baseline, coefficients. Add a short results section to README.
- Optional stretch: add 2–3 more features (e.g. rolling offensive/defensive rating,
  back-to-back flag) and show whether they help on the time-split test.

**Phase gate:** owner reviews results and README before Phase 2 begins.

## Phase 2 — Shot quality model (xFG%)
- Pull shot-chart data via `nba_api` (`shotchartdetail`), several seasons, cached.
- Features: shot distance, x/y location and angle, shot type, period, shot-clock context
  if available, score context. NOTE: defender-distance fields may not be available in the
  public API for recent seasons — check what actually exists, use what does, and document
  the limitation in the README instead of faking it.
- Model: XGBoost classifier. Evaluate with log loss, ROC-AUC, and a calibration plot
  against the naive baseline. League-average FG% is ~46–47%; a good public-data model
  gets meaningful but modest lift.
- Deliverable: notebook with a shot-chart visualization colored by predicted make
  probability, plus README section.

**Phase gate:** owner reviews before Phase 3.

## Phase 3 — Win probability model + dashboard
- Pull play-by-play logs (`playbyplayv2` or newer endpoint), several seasons, cached.
  This data is large: build the parser incrementally, test on one game first.
- Features per event: score margin, seconds remaining, period, possession, bonus/foul
  state, timeouts remaining if derivable.
- Model: small PyTorch feed-forward net (a logistic baseline first for comparison).
  Output: P(home win) at each moment. Evaluate with log loss over held-out season and
  calibration by game-time bucket.
- Dashboard: **Streamlit first**, replaying a historical game with a live-updating win
  probability curve. Polling is fine. WebSockets/live games are a stretch goal only
  after the replay dashboard works. Produce a GIF of the replay for the README.

## Working style each session
1. State what you're about to do in a sentence or two, then do it.
2. Run the code you change. Show the actual output.
3. If something is ambiguous or a dependency/API has changed, say so and pick the
   simplest honest option rather than adding complexity.
4. End each session by listing: what got done, what's next, any open questions for the
   owner to review here or discuss with chat Claude.
