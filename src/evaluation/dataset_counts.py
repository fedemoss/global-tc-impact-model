"""Event, cyclone and ADM1 counts, as built and as evaluated."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import INPUT_DIR, OUTPUT_DIR

SUFFIX = "_v2"
OUT = OUTPUT_DIR / "model_output"


def show(name, df, ev="DisNo.", sid="sid"):
    adm1 = None
    if {"GID_0", "GID_1"} <= set(df.columns):
        adm1 = df[["GID_0", "GID_1"]].astype(str).drop_duplicates().shape[0]
    print(f"{name:<34} {df[ev].nunique():>6} {df[sid].nunique():>8} "
          f"{(df['GID_0'].nunique() if 'GID_0' in df else 0):>6} "
          f"{(adm1 if adm1 is not None else 0):>7} {len(df):>12,}")


print("=" * 90)
print(f"{'':<34} {'events':>6} {'cyclones':>8} {'countr':>6} {'ADM1':>7} {'rows':>12}")
print("=" * 90)

# 1. the dataset as built
cols = ["DisNo.", "sid", "GID_0", "GID_1", "year", "perc_affected_pop_grid_region",
        "Total Affected", "level"]
raw = pd.read_parquet(INPUT_DIR / "model_input_dataset" / "training_dataset.parquet",
                      columns=cols)
for c in ("DisNo.", "sid", "GID_0", "GID_1", "level"):
    raw[c] = raw[c].astype(str)
show("input dataset (grid cells)", raw)
print(f"{'':<34} years {int(raw['year'].min())}-{int(raw['year'].max())}")

# 2. what the model evaluates, per run
runs = {
    "global (LOOCV)": f"2-stage-XGBoost_global_adm1{SUFFIX}",
    "walk-forward": f"2-stage-XGBoost_walk_forward_adm1{SUFFIX}",
    "basin-constrained": f"2-stage-XGBoost_geo_constrained_adm1{SUFFIX}",
    "ADM1-fitted": f"2-stage-XGBoost_global_adm1_trained{SUFFIX}",
    "no prev. events": f"2-stage-XGBoost_global_adm1_no_N_events_5_years{SUFFIX}",
}
look = raw[["DisNo.", "sid"]].drop_duplicates()

print()
for label, folder in runs.items():
    files = sorted((OUT / folder).glob("predictions_event_*.csv"))
    if not files:
        print(f"{label:<34} not scored")
        continue
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in
                                ("DisNo.", "GID_0", "GID_1", "sid"))
                    for f in files], ignore_index=True)
    df["DisNo."] = df["DisNo."].astype(str)
    if "sid" not in df.columns or df["sid"].isna().any():
        df = df.drop(columns=["sid"], errors="ignore").merge(look, on="DisNo.", how="left")
    show(label, df)

# 3. the shape of an ADM1-event panel
g = pd.concat([pd.read_csv(f, usecols=lambda c: c in ("DisNo.", "GID_0", "GID_1"))
               for f in sorted((OUT / runs["global (LOOCV)"]).glob("predictions_event_*.csv"))],
              ignore_index=True)
g["DisNo."] = g["DisNo."].astype(str)
g = g.merge(look, on="DisNo.", how="left")
per_ev = g.groupby("DisNo.").size()
per_cyc = g.groupby("sid").size()
ev_per_cyc = g.groupby("sid")["DisNo."].nunique()

print("\n" + "=" * 90)
print("ADM1 observations per unit of analysis (global run)")
print("=" * 90)
print(f"  ADM1-event observations      {len(g):,}")
print(f"  per country-event            median {per_ev.median():.0f}, "
      f"mean {per_ev.mean():.1f}, max {per_ev.max()}")
print(f"  per cyclone                  median {per_cyc.median():.0f}, "
      f"mean {per_cyc.mean():.1f}, max {per_cyc.max()}")
print(f"  country-events per cyclone   1: {(ev_per_cyc == 1).sum()}, "
      f">1: {(ev_per_cyc > 1).sum()}, max {ev_per_cyc.max()}")

print("\n  impact class balance (reported):")
rep = pd.concat([pd.read_csv(f, usecols=lambda c: c in ("reported",))
                 for f in sorted((OUT / runs["global (LOOCV)"]).glob("predictions_event_*.csv"))],
                ignore_index=True)["reported"]
print(f"    no impact (=0)             {(rep <= 0).sum():>7,}  ({100*(rep <= 0).mean():.1f}%)")
print(f"    low impact (0-15%]         {((rep > 0) & (rep <= 15)).sum():>7,}  "
      f"({100*((rep > 0) & (rep <= 15)).mean():.1f}%)")
print(f"    high impact (>15%)         {(rep > 15).sum():>7,}  ({100*(rep > 15).mean():.1f}%)")
