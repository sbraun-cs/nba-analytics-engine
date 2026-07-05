"""Phase 3 - recalibration experiment for the early-game home-bias.

The core/prior model over-predicts the home team early (Q1 ~5 pts too high). The
error analysis traced this to an INTERCEPT problem: home-court advantage was
higher in the training seasons than in the 2024-25 test season (the decline shown
in Phase 1), so the model's baseline is too home-friendly.

Fix (chosen method, justified): a single log-odds intercept shift, fit so the
model's average home-win prediction matches the empirical home-win rate of the
most recent *training* season (2023-24), whose home advantage already matches the
2024-25 era. This is leak-free (calibrated on train data, never the test set),
one parameter (negligible overfitting), and a constant logit shift barely moves
the saturated late-game predictions - so it fixes early calibration without
disturbing the endgame.

    python -m src.phase3.recalibrate
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import log_loss

from src.phase3.baseline import (
    ALL_FEATURES, build_dataset, calibration_by_period, fit_logistic, sampled_split,
)

RECENT_SEASON_CODE = "23"  # game_id[3:5] for 2023-24, the most recent train season


def _apply_delta(p: np.ndarray, delta: float) -> np.ndarray:
    """Shift probabilities by a constant in log-odds space."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(logit + delta)))


def fit_intercept_shift(p_calib: np.ndarray, y_calib: np.ndarray) -> float:
    """Find the log-odds shift that matches mean prediction to the base rate."""
    target = float(np.mean(y_calib))
    return brentq(lambda d: float(_apply_delta(p_calib, d).mean()) - target, -6, 6)


def main():
    train_ids, test_ids = sampled_split()
    train = build_dataset(train_ids)
    test = build_dataset(test_ids)

    model = fit_logistic(train, ALL_FEATURES)
    p_train = model.predict_proba(train[ALL_FEATURES])[:, 1]

    recent = (train["game_id"].str[3:5] == RECENT_SEASON_CODE).to_numpy()
    delta = fit_intercept_shift(p_train[recent], train["home_win"].to_numpy()[recent])

    p_test = model.predict_proba(test[ALL_FEATURES])[:, 1]
    p_recal = _apply_delta(p_test, delta)

    print(f"intercept shift delta = {delta:+.3f}  "
          f"(fit on {recent.sum():,} events from the 2023-24 season)")
    print(f"overall log loss:  before {log_loss(test['home_win'], p_test):.4f}  "
          f"->  after {log_loss(test['home_win'], p_recal):.4f}")

    cal = calibration_by_period(test, p_test).rename(columns={"mean_pred": "pred_before"})
    cal["pred_after"] = calibration_by_period(test, p_recal)["mean_pred"]
    print("\nCalibration by period (mean predicted vs actual home-win rate):")
    print(cal[["events", "actual", "pred_before", "pred_after"]].round(4).to_string())


if __name__ == "__main__":
    main()
