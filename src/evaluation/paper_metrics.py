"""Every scored model, at ADM1 level, with cyclone-clustered intervals.

Writes a CSV of the full numbers and a LaTeX body for each table, so the
figures in the paper and the numbers beside them come from one computation.
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import OUTPUT_DIR
from src.evaluation.bootstrap import bootstrap_binary_metrics

# A re-run kept under its own folder name writes its own metrics beside
# the previous run's, rather than replacing them.
SUFFIX = os.environ.get("TC_RUN_SUFFIX", "")
OUT = OUTPUT_DIR / f"metrics{SUFFIX}"
N_BOOT = 1000
EVENT_COL = "sid"
THRESHOLDS = (0, 15)

# label -> output folder, grouped by the comparison it belongs to
GROUPS = {
    "models": {
        "Historical": "historical_global_adm1",
        "Windspeed-exposed": "windspeed-exposed_global_adm1",
        "Windspeed-historical": "windspeed-historical_global_adm1",
        "2-stg-XGBoost": "2-stage-XGBoost_global_adm1",
    },
    "strategies": {
        "LOOCV": "2-stage-XGBoost_global_adm1",
        "Walk-forward": "2-stage-XGBoost_walk_forward_adm1",
        "Basin-constrained": "2-stage-XGBoost_geo_constrained_adm1",
    },
    "variants": {
        "Grid-fitted": "2-stage-XGBoost_global_adm1",
        "ADM1-fitted": "2-stage-XGBoost_global_adm1_trained",
        "No previous events": "2-stage-XGBoost_global_adm1_no_N_events_5_years",
    },
}
REPORT = ["precision", "recall", "specificity", "accuracy", "f1", "csi"]


def load(folder):
    # The rule baselines are not re-run with the model and keep a single
    # set of outputs; every fitted variant follows the run.
    if folder.startswith("2-stage-XGBoost"):
        folder += SUFFIX
    base = OUTPUT_DIR / "model_output" / folder
    bundle = base.with_suffix(".parquet")
    if bundle.exists():
        df = pd.read_parquet(bundle)
    else:
        files = sorted(base.glob("predictions_event_*.csv"))
        if not files:
            return None
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # A run fitted at ADM1 is not aggregated, so it carries the model's own
    # column names; a grid run's aggregation renames them. Both are a reported
    # and a predicted share of population.
    if "reported" not in df.columns and "perc_affected_pop_grid_region" in df.columns:
        df = df.copy()
        df["reported"] = df["perc_affected_pop_grid_region"].clip(0, 100)
        df["predicted"] = df["prediction_perc"].clip(0, 100)
    return df.dropna(subset=["reported", "predicted"])


def with_sid(df, lookup):
    if EVENT_COL in df.columns and df[EVENT_COL].notna().all():
        return df
    return df.drop(columns=[EVENT_COL], errors="ignore").merge(
        lookup, on="DisNo.", how="left")


def latex_rows(table, order):
    """One row per model, each metric as its estimate over its interval."""
    lines = []
    for t in THRESHOLDS:
        for label in order:
            r = table[(table["model"] == label) & (table["threshold"] == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = []
            for m in REPORT:
                cells.append(
                    f"\\makecell{{{r[m]:.2f}\\\\\\scriptsize"
                    f"{{({r[m + '_ci_low']:.2f}, {r[m + '_ci_high']:.2f})}}}}")
            lines.append(f"{label} & {t} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # the storm identifier, for clustering; some outputs no longer carry it
    lookup = None
    for folder in ("2-stage-XGBoost_global_adm1",):
        d = load(folder)
        if d is not None and EVENT_COL in d.columns:
            lookup = d[["DisNo.", EVENT_COL]].drop_duplicates()
    if lookup is None:
        raise SystemExit("no output carries the storm identifier")

    everything = []
    for group, members in GROUPS.items():
        frames, rows = {}, []
        for label, folder in members.items():
            df = load(folder)
            if df is None:
                print(f"  {group}/{label}: not scored")
                continue
            frames[label] = with_sid(df, lookup)

        if len(frames) > 1:
            # compare on the units every member scored
            keys = ["DisNo.", "GID_1"]
            shared = set.intersection(*[
                set(map(tuple, f[keys].astype(str).values)) for f in frames.values()])
            frames = {k: f[[tuple(r) in shared
                            for r in f[keys].astype(str).values]]
                      for k, f in frames.items()}
            print(f"\n{group}: {len(frames)} members on {len(shared):,} shared units")

        for label, df in frames.items():
            m = bootstrap_binary_metrics(
                df, event_col=EVENT_COL, thresholds=THRESHOLDS,
                target_col="reported", pred_col="predicted",
                model_name=label, n_boot=N_BOOT, random_state=42)
            m["group"] = group
            m["events"] = df["DisNo."].nunique()
            m["units"] = len(df)
            rows.append(m)

        if not rows:
            continue
        table = pd.concat(rows, ignore_index=True)
        everything.append(table)

        print(f"\n=== {group} ===")
        show = table[["model", "threshold", "events", "units"] + REPORT
                     + ["mean_abs_error"]].round(4)
        print(show.to_string(index=False))

        tex = OUT / f"table_{group}.tex"
        tex.write_text(latex_rows(table, list(members)))
        print(f"  -> {tex}")

    full = pd.concat(everything, ignore_index=True)
    csv = OUT / "paper_metrics_all.csv"
    full.to_csv(csv, index=False)
    print(f"\nwrote {csv}  ({len(full)} rows)")


if __name__ == "__main__":
    main()
