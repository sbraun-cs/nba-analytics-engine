"""Phase 2 extension: 20 years of NBA shot selection (2004-2024).

Uses the local historical shot archive (21 season CSVs, ~4.2M shots) to chart the
league's shift away from the mid-range toward threes and shots at the rim - the
defining change in modern NBA shot selection. Reads only local CSVs; nothing is
fetched or committed.

Expected layout:
    data/shots_archive/NBA_2004_Shots.csv ... NBA_2024_Shots.csv

Run:
    python -m src.phase2.shot_evolution
Outputs:
    docs/shot_evolution.png   (the chart for the README)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data" / "shots_archive"
OUT = ROOT / "docs" / "shot_evolution.png"

# Map the archive's BASIC_ZONE values into three interpretable buckets.
AT_RIM = {"Restricted Area"}
MIDRANGE = {"Mid-Range", "In The Paint (Non-RA)"}
# Everything with a 3PT shot type is a three; handled explicitly below.


def classify(df: pd.DataFrame) -> pd.Series:
    """Return a Series bucketing each shot into 'Three', 'At rim', or 'Mid-range'."""
    is_three = df["SHOT_TYPE"].str.contains("3PT", na=False)
    zone = df["BASIC_ZONE"].fillna("")
    out = pd.Series("Mid-range", index=df.index)
    out[zone.isin(AT_RIM)] = "At rim"
    out[is_three] = "Three"  # three-point classification wins over zone
    return out


def season_mix(csv: Path) -> dict:
    """Fraction of shots in each bucket for one season file."""
    # Only the two columns we need, to keep 4.2M rows light on memory.
    df = pd.read_csv(csv, usecols=["SHOT_TYPE", "BASIC_ZONE"])
    buckets = classify(df)
    frac = buckets.value_counts(normalize=True)
    year = int(csv.stem.split("_")[1])
    return {
        "season": year,
        "Three": frac.get("Three", 0.0) * 100,
        "Mid-range": frac.get("Mid-range", 0.0) * 100,
        "At rim": frac.get("At rim", 0.0) * 100,
    }


def build() -> pd.DataFrame:
    files = sorted(ARCHIVE.glob("NBA_*_Shots.csv"))
    if not files:
        raise SystemExit(
            f"No shot CSVs found in {ARCHIVE}.\n"
            "Put the archive season files (NBA_2004_Shots.csv ...) there first."
        )
    rows = []
    for f in files:
        row = season_mix(f)
        rows.append(row)
        print(f"  {row['season']}: "
              f"3PT {row['Three']:.1f}%  mid {row['Mid-range']:.1f}%  rim {row['At rim']:.1f}%")
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def plot(df: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"Three": "#e8794c", "Mid-range": "#4c9be8", "At rim": "#2ecc71"}
    for col in ["Three", "Mid-range", "At rim"]:
        ax.plot(df["season"], df[col], marker="o", ms=4, lw=2.4,
                color=colors[col], label=col)
    ax.set_xlabel("Season")
    ax.set_ylabel("Share of all field-goal attempts (%)")
    ax.set_title("Twenty years of NBA shot selection: the three rises, the mid-range falls",
                 fontsize=13, weight="bold")
    ax.legend(frameon=True, loc="center left")
    ax.set_xticks(df["season"][::2])
    plt.xticks(rotation=45)
    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"\nSaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    df = build()
    plot(df)
    # Print the headline deltas for the README.
    first, last = df.iloc[0], df.iloc[-1]
    print(f"\n{first['season']} -> {last['season']}:")
    print(f"  Three     {first['Three']:.1f}% -> {last['Three']:.1f}%")
    print(f"  Mid-range {first['Mid-range']:.1f}% -> {last['Mid-range']:.1f}%")
    print(f"  At rim    {first['At rim']:.1f}% -> {last['At rim']:.1f}%")
