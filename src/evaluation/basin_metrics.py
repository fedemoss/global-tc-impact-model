"""Performance per cyclone basin, for every scored model.

The basin belongs to the storm, not to the country, so it is taken from the
dataset's per-sid value and attached to the predictions through the storm id.
Intervals resample cyclones within the basin, so a basin carried by a handful
of storms reports an interval wide enough to say so.
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import INPUT_DIR, OUTPUT_DIR
from src.evaluation.bootstrap import bootstrap_binary_metrics

SUFFIX = os.environ.get("TC_RUN_SUFFIX", "")
OUT = OUTPUT_DIR / f"metrics{SUFFIX}"
N_BOOT = 1000
THRESHOLDS = (0, 15)
MIN_CYCLONES = 2          # below this a bootstrap over cyclones says nothing

FOLDERS = {
    "Historical": "historical_global_adm1",
    "Windspeed-exposed": "windspeed-exposed_global_adm1",
    "Windspeed-historical": "windspeed-historical_global_adm1",
    "LOOCV": "2-stage-XGBoost_global_adm1",
    "Walk-forward": "2-stage-XGBoost_walk_forward_adm1",
    "Basin-constrained": "2-stage-XGBoost_geo_constrained_adm1",
    "ADM1-fitted": "2-stage-XGBoost_global_adm1_trained",
    "No previous events": "2-stage-XGBoost_global_adm1_no_N_events_5_years",
}
REPORT = ["precision", "recall", "f1", "csi", "mean_abs_error"]


def load(folder):
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
    if "reported" not in df.columns and "perc_affected_pop_grid_region" in df.columns:
        df = df.copy()
        df["reported"] = df["perc_affected_pop_grid_region"].clip(0, 100)
        df["predicted"] = df["prediction_perc"].clip(0, 100)
    return df.dropna(subset=["reported", "predicted"])


def basin_lookup():
    """DisNo. -> (sid, cyclone_basin), from the dataset the models were fitted on."""
    path = INPUT_DIR / "model_input_dataset" / "training_dataset.parquet"
    d = pd.read_parquet(path, columns=["DisNo.", "sid", "cyclone_basin"])
    d = d.drop_duplicates()
    for c in d.columns:
        d[c] = d[c].astype(str)
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    look = basin_lookup()
    print(f"{look['sid'].nunique()} cyclones over basins: "
          f"{sorted(look['cyclone_basin'].unique())}\n")

    rows = []
    for label, folder in FOLDERS.items():
        df = load(folder)
        if df is None:
            print(f"  {label}: not scored")
            continue
        df = df.copy()
        df["DisNo."] = df["DisNo."].astype(str)
        df = df.drop(columns=[c for c in ("sid", "cyclone_basin") if c in df.columns])
        df = df.merge(look, on="DisNo.", how="left")

        missing = df["cyclone_basin"].isna().sum()
        if missing:
            print(f"  {label}: {missing} rows without a basin, dropped")
            df = df.dropna(subset=["cyclone_basin"])

        for basin, g in df.groupby("cyclone_basin"):
            n_cyc = g["sid"].nunique()
            if n_cyc < MIN_CYCLONES:
                print(f"  {label}/{basin}: {n_cyc} cyclone, skipped")
                continue
            m = bootstrap_binary_metrics(
                g, event_col="sid", thresholds=THRESHOLDS,
                target_col="reported", pred_col="predicted",
                model_name=label, n_boot=N_BOOT, random_state=42)
            m["basin"] = basin
            m["cyclones"] = n_cyc
            m["events"] = g["DisNo."].nunique()
            m["units"] = len(g)
            rows.append(m)

    table = pd.concat(rows, ignore_index=True)
    csv = OUT / "basin_metrics.csv"
    table.to_csv(csv, index=False)

    pd.set_option("display.width", 200)
    for t in THRESHOLDS:
        print(f"\n{'=' * 90}\nthreshold {t}\n{'=' * 90}")
        sub = table[table["threshold"] == t]
        show = sub.pivot_table(index=["basin", "cyclones", "events", "units"],
                               columns="model", values="f1").round(4)
        print(show.to_string())
        print(f"\n-- {'; '.join(REPORT)} for the main model --")
        main_rows = sub[sub["model"] == "LOOCV"]
        print(main_rows[["basin", "cyclones", "events", "units"] + REPORT
                        + ["f1_ci_low", "f1_ci_high"]].round(4).to_string(index=False))

    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
