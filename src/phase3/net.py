"""Phase 3 - PyTorch feed-forward net: does a net beat logistic-plus-prior?

Same features (core + Phase 1 prior) and same game/season split as the logistic
model, so the comparison is apples-to-apples. Either answer is a finding.

Finding: the net does NOT beat logistic-plus-prior (log loss ~0.468 vs 0.423,
AUC ~0.861 vs 0.884). Unregularized it overfits badly (test log loss diverges as
it trains); across dropout/weight-decay settings it plateaus near the logistic
*core* level. The win-probability relationship is close to linear in log-odds, so
a well-regularized logistic model is the better tool here.

    python -m src.phase3.net
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.phase3.baseline import (
    ALL_FEATURES, FEATURES, build_dataset, fit_logistic, sampled_split,
)

SEED = 0


class WinProbNet(nn.Module):
    """Small MLP: a couple of hidden layers with ReLU + dropout, logit output."""

    def __init__(self, n_in: int, hidden=(64, 32), p_drop: float = 0.3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _split_by_game(df: pd.DataFrame, frac_val: float = 0.15, seed: int = SEED):
    """Hold out whole games for validation (game-disjoint, like the main split)."""
    games = df["game_id"].drop_duplicates()
    val_games = set(games.sample(frac=frac_val, random_state=seed))
    val = df[df["game_id"].isin(val_games)]
    fit = df[~df["game_id"].isin(val_games)]
    return fit, val


def train_net(train_df, features, epochs=200, patience=12, seed=SEED):
    """Train with early stopping on a game-disjoint validation slice."""
    torch.manual_seed(seed)
    fit, val = _split_by_game(train_df, seed=seed)
    scaler = StandardScaler().fit(fit[features])

    def tensors(frame):
        X = torch.tensor(scaler.transform(frame[features]), dtype=torch.float32)
        y = torch.tensor(frame["home_win"].to_numpy(), dtype=torch.float32)
        return X, y

    Xf, yf = tensors(fit)
    Xv, yv = tensors(val)

    model = WinProbNet(len(features))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val, best_state, bad = float("inf"), None, 0
    n, batch = len(Xf), 4096
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(Xf[idx]), yf[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv), yv).item()
        if vloss < best_val - 1e-4:
            best_val, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, scaler


def net_prob(model, scaler, df, features) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X = torch.tensor(scaler.transform(df[features]), dtype=torch.float32)
        return torch.sigmoid(model(X)).numpy()


def main():
    train_ids, test_ids = sampled_split()
    print(f"Building datasets ({len(train_ids)} train + {len(test_ids)} test games)...")
    train = build_dataset(train_ids)
    test = build_dataset(test_ids)

    rows = []
    for label, feats in [("Logistic (core)", FEATURES),
                         ("Logistic (core+prior)", ALL_FEATURES)]:
        model = fit_logistic(train, feats)
        p = model.predict_proba(test[feats])[:, 1]
        rows.append((label, log_loss(test["home_win"], p), roc_auc_score(test["home_win"], p)))

    net, scaler = train_net(train, ALL_FEATURES)
    p = net_prob(net, scaler, test, ALL_FEATURES)
    rows.append(("PyTorch net (core+prior)",
                 log_loss(test["home_win"], p), roc_auc_score(test["home_win"], p)))

    print(f"\n{'model':<28}{'log loss':>10}{'ROC-AUC':>10}")
    for label, ll, auc in rows:
        print(f"{label:<28}{ll:>10.4f}{auc:>10.4f}")


if __name__ == "__main__":
    main()
