"""Paired cyclone bootstrap of the difference between two scored models.

Table 1 gives each model its own interval. Whether two models differ is a
separate question, and overlapping intervals do not answer it: the two models
are scored on the same storms, so their errors are correlated, and the
difference is far less uncertain than either score. Here both models are
scored on the same resampled cyclones in every replicate, and the interval is
the one of the difference itself (model A minus model B).

Usage
-----
    python src/evaluation/model_comparison.py
    python src/evaluation/model_comparison.py --a 2-stage-XGBoost_global_adm1 \
        --b windspeed-exposed_global_adm1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.bootstrap import compute_metrics_at_threshold
from src.evaluation.paper_metrics import EVENT_COL, OUT, THRESHOLDS, load, with_sid

METRICS = ("precision", "recall", "f1", "csi", "mean_abs_error", "rmse", "qwk")
UNIT_KEYS = ["DisNo.", "GID_1"]


def paired_frame(df_a, df_b):
    """One row per unit both models scored, with both predictions side by side."""
    a = df_a[UNIT_KEYS + [EVENT_COL, "reported", "predicted"]].copy()
    b = df_b[UNIT_KEYS + ["predicted"]].copy()
    for frame in (a, b):
        frame[UNIT_KEYS] = frame[UNIT_KEYS].astype(str)
    paired = a.merge(b, on=UNIT_KEYS, how="inner", suffixes=("_a", "_b"))
    if paired.duplicated(UNIT_KEYS).any():
        raise ValueError("a unit appears more than once in one of the two outputs")
    return paired


def _both(df, threshold):
    out = {}
    for side in ("a", "b"):
        m = compute_metrics_at_threshold(df, threshold, target_col="reported",
                                         pred_col=f"predicted_{side}",
                                         nan_on_undefined=True)
        out[side] = m
    return out


def paired_bootstrap(paired, thresholds=THRESHOLDS, n_boot=1000, ci=95, random_state=42):
    """Point difference, interval and share of replicates favouring A, per metric."""
    rng = np.random.default_rng(random_state)
    events = paired[EVENT_COL].unique()
    groups = {e: g for e, g in paired.groupby(EVENT_COL)}
    alpha = (100 - ci) / 2
    rows = []
    for threshold in thresholds:
        point = _both(paired, threshold)
        diffs = {k: [] for k in METRICS}
        for _ in range(n_boot):
            sample = pd.concat([groups[e] for e in rng.choice(events, len(events))],
                               ignore_index=True)
            m = _both(sample, threshold)
            for k in METRICS:
                d = m["a"][k] - m["b"][k]
                if not pd.isna(d):
                    diffs[k].append(d)
        for k in METRICS:
            values = np.asarray(diffs[k], dtype="float64")
            # For an error metric a negative difference favours A.
            better = values < 0 if k in ("mean_abs_error", "rmse") else values > 0
            rows.append({
                "threshold": threshold, "metric": k,
                "a": point["a"][k], "b": point["b"][k],
                "difference": point["a"][k] - point["b"][k],
                "ci_low": float(np.percentile(values, alpha)) if values.size else np.nan,
                "ci_high": float(np.percentile(values, 100 - alpha)) if values.size else np.nan,
                "share_favouring_a": float(better.mean()) if values.size else np.nan,
                "n_valid_boot": int(values.size),
            })
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--a", default="2-stage-XGBoost_global_adm1", help="output folder of model A")
    p.add_argument("--b", default="2-stage-GLM_global_adm1", help="output folder of model B")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args(argv)

    df_a, df_b = load(args.a), load(args.b)
    for name, df in ((args.a, df_a), (args.b, df_b)):
        if df is None:
            raise SystemExit(f"no scored output for {name}")
    lookup = df_a[["DisNo.", EVENT_COL]].drop_duplicates() if EVENT_COL in df_a.columns else \
        df_b[["DisNo.", EVENT_COL]].drop_duplicates()
    paired = paired_frame(with_sid(df_a, lookup), with_sid(df_b, lookup))
    if paired[EVENT_COL].isna().any():
        raise SystemExit(f"some units have no {EVENT_COL}; cannot resample cyclones")
    print(f"{args.a} vs {args.b}: {len(paired):,} shared units, "
          f"{paired[EVENT_COL].nunique()} cyclones")

    table = paired_bootstrap(paired, n_boot=args.n_boot, random_state=args.random_state)
    print(table.round(4).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"comparison_{args.a}_vs_{args.b}.csv"
    table.to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
