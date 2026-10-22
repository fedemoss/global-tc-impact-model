# Figures — v2 run

Written by `metrics_overview.ipynb` with `TC_RUN_SUFFIX=_v2`, from the model
output folders carrying that suffix. The figures in `../figs/` come from the
first run and are kept beside these, so the two can be compared rather than
one replacing the other.

What distinguishes this run:

- **Folds withhold the whole cyclone.** Each fold still predicts one
  country-event, the level impact is reported at, but its training set drops
  every record of the storm under test. A cyclone making landfall in several
  countries is one EM-DAT record per country, and 162 of 541 cyclones here do
  so, which changes the training set of 423 of 802 folds.
- **Stage-1 threshold 0.60**, from the exhaustive ratios search
  (`u1=2`, `u2=10`), against 0.70 in the first run.

Scoring the same predictions at 0.70 shows the fold correction accounts for
almost none of the difference between the two runs: f1 moves by 0.0005 at the
reporting threshold. The visible change is the threshold, not the folds.
