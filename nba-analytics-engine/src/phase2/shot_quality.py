"""Phase 2 - Shot quality model (expected FG%, "xFG%").

Predicts the probability that a field-goal attempt goes in, from where and how
it was taken. Trains XGBoost on several seasons of league-wide shot-chart data
and evaluates against the naive league-average baseline.

What the public `shotchartdetail` endpoint gives us, and what it does NOT:
  available : shot distance, x/y location (-> angle), 2PT/3PT, action type
              (layup/jumper/dunk/...), shot zone, period, game clock.
  MISSING   : defender distance, shot-clock, and score margin. These would be
              the most valuable "shot difficulty" signals, but they are not in
              the public endpoint for these seasons, so the model cannot use
              them. This is documented, not faked (see CLAUDE.md / README).

Run directly:  python -m src.phase2.shot_quality
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

from src.ingest.nba_data import DATA_DIR, recent_seasons, shot_charts

# --- configuration -----------------------------------------------------------
N_SEASONS = 5
LAST_SEASON_START = 2024          # -> most recent season is 2024-25
N_TEST_SEASONS = 1               # hold out the final season for testing
TOP_ACTION_TYPES = 15            # keep the N most common shot actions, rest -> "Other"

NUMERIC_FEATURES = [
    "shot_distance", "loc_x", "loc_y", "shot_angle",
    "is_three", "period", "time_remaining_sec",
]
CATEGORICAL_FEATURES = ["SHOT_ZONE_BASIC", "action_grouped"]


# --- feature engineering -----------------------------------------------------
def build_features(seasons: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return (dataframe with engineered features + target + season, feature_cols)."""
    df = shot_charts(seasons)

    # Basic numeric features straight from the endpoint.
    df["shot_distance"] = df["SHOT_DISTANCE"].astype(float)
    df["loc_x"] = df["LOC_X"].astype(float)
    df["loc_y"] = df["LOC_Y"].astype(float)
    df["is_three"] = (df["SHOT_TYPE"] == "3PT Field Goal").astype(int)
    df["period"] = df["PERIOD"].astype(int)
    df["time_remaining_sec"] = (
        df["MINUTES_REMAINING"].astype(int) * 60 + df["SECONDS_REMAINING"].astype(int)
    )

    # Angle from straight-on (0 deg = facing the rim, 90 deg = along the baseline).
    df["shot_angle"] = np.degrees(np.arctan2(df["loc_x"].abs(), df["loc_y"].clip(lower=0)))

    # ACTION_TYPE is high-cardinality; keep the most common, bucket the rest.
    top = df["ACTION_TYPE"].value_counts().head(TOP_ACTION_TYPES).index
    df["action_grouped"] = df["ACTION_TYPE"].where(df["ACTION_TYPE"].isin(top), "Other")

    df["target"] = df["SHOT_MADE_FLAG"].astype(int)

    # One-hot the categoricals. get_dummies on the full frame keeps train/test
    # columns aligned; it uses only feature values, never the target.
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    feats = pd.concat([df[NUMERIC_FEATURES + ["target", "SEASON"]], dummies], axis=1)

    feature_cols = NUMERIC_FEATURES + list(dummies.columns)
    return feats.dropna(subset=feature_cols).reset_index(drop=True), feature_cols


# --- evaluation --------------------------------------------------------------
def time_split(df: pd.DataFrame, n_test_seasons: int):
    """Train on earlier seasons, test on the last `n_test_seasons`."""
    seasons = sorted(df["SEASON"].unique())
    test_seasons = seasons[-n_test_seasons:]
    is_test = df["SEASON"].isin(test_seasons)
    return df[~is_test].copy(), df[is_test].copy(), test_seasons


def train_model(train: pd.DataFrame, feature_cols: list[str]) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=0,
    )
    model.fit(train[feature_cols], train["target"])
    return model


def evaluate(model, train, test, feature_cols):
    prob = model.predict_proba(test[feature_cols])[:, 1]

    ll = log_loss(test["target"], prob)
    auc = roc_auc_score(test["target"], prob)

    # Naive baseline: predict the training league-average make rate for every shot.
    base_rate = train["target"].mean()
    base_ll = log_loss(test["target"], np.full(len(test), base_rate))

    print("\n=== Shot quality model (XGBoost) ===")
    print(f"  train shots        : {len(train):,}")
    print(f"  test shots         : {len(test):,}")
    print(f"  league avg (train) : {base_rate:.4f}")
    print(f"  log loss           : {ll:.4f}   (baseline {base_ll:.4f})")
    print(f"  ROC-AUC            : {auc:.4f}   (baseline 0.5000)")

    imp = pd.Series(model.feature_importances_, index=feature_cols)
    print("  top features (gain-weighted importance):")
    for name, val in imp.sort_values(ascending=False).head(8).items():
        print(f"    {name:<28} {val:.4f}")

    return {"log_loss": ll, "roc_auc": auc, "baseline_log_loss": base_ll,
            "league_avg": base_rate}


def main():
    seasons = recent_seasons(N_SEASONS, LAST_SEASON_START)
    print(f"Seasons: {seasons[0]} ... {seasons[-1]}  ({len(seasons)} seasons)")

    feats, feature_cols = build_features(seasons)
    print(f"Modeling {len(feats):,} shots with {len(feature_cols)} features.")

    # Cache processed features + predictions so the notebook can draw shot charts
    # without refetching or retraining from scratch.
    train, test, test_seasons = time_split(feats, N_TEST_SEASONS)
    print(f"Train shots: {len(train):,} | Test shots: {len(test):,} "
          f"| Test season: {test_seasons}")

    model = train_model(train, feature_cols)
    evaluate(model, train, test, feature_cols)

    print("\nReminder: without defender distance / shot clock, AUC in the ~0.65-0.72 "
          "range is expected for public data. Much higher would be suspicious.")


if __name__ == "__main__":
    main()
