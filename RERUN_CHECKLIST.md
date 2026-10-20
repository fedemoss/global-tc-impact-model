# Rerun checklist

Updated 20 Oct 2026 against main `6660c4c` and the October runs. Closed items
stay listed with how they were closed; the open ones are what stands between
the current runs and numbers the manuscript can freeze.

---

## 0. Merge this branch first — the October runs used the leaky unit again

The October refactor was built from `main`, which never received this branch's
default flip, so `event_col="DisNo."` was again the default in
`train.py` / `cv_strategies.py` / `bootstrap.py` / the metrics CLI, and
`python main.py --run-models` passes nothing. Verified on a synthetic corpus
with six two-country storms: the default path produced 30 folds, not 24.
Meanwhile the hyperparameter search *is* sid-grouped — so the u1=2 / u2=10 /
0.70 configuration was tuned under one grouping and the reported LOOCV and
walk-forward numbers evaluated under another.

- [ ] Merge this branch into `main` (it is rebuilt on top of `6660c4c`;
      sid is the default everywhere again, `DisNo.` stays available as an
      explicit argument writing to `_by_DisNo.` directories).
- [ ] **Relaunch LOOCV, walk-forward, and the mldevbox geo-constrained and
      ablation runs** under the merged defaults. Any run launched through
      `main.py` before the merge is DisNo.-grouped and cannot be quoted.
- [ ] The `\revii{}` numbers going into the manuscript must come from the
      sid-grouped runs — the revision text explicitly claims cyclone-grouped
      evaluation.

## 1. Provenance of each quoted run (confirm)

- [ ] **Folds = physical cyclones.** The LOOCV log prints
      `N physical cyclones / M country-event records / rows -> K folds`;
      confirm K equals the cyclone count, not the record count.
- [ ] **Which commit ran.** Record the hash in the run directory
      (`git rev-parse HEAD > <run_dir>/COMMIT`).
- [ ] **Which specification ran** — with or without `N_events_5_years`?
      The with/without pair is the ablation; the primary-specification
      decision is still open with KK.

## 2. The search selected boundary values — re-run on the widened grids

u2 = 10 and clf_threshold = 0.70 both sit at the **edge** of the previous
search space, the classic sign of a censored optimum. This branch widens the
grids (u1/u2 up to 20, threshold up to 0.80).

- [ ] Re-run the search on the widened grids (after the merge, so search and
      evaluation share the sid unit):

      ```bash
      python test/hyperparameter_search.py --n-iter 80 --n-events 200
      ```

- [ ] If the selection moves interior, the old values were censored; if it
      stays at u2=10 / 0.70, say so in the appendix — that is a fine answer.
- [ ] Keep the written JSON with the run; it records `group_col` and the
      search/hold-out sid lists.

## 3. Attribution runs

"How much of the metric movement is corrected evaluation vs re-tuned model vs
corrected data?" One extra run splits the biggest factors — the round-1 fixed
configuration under the new grouping and data:

```python
from src.models.train import execute_training_run
execute_training_run("2-stage-XGBoost", "global",
                     use_tuned=False)   # hook restored on this branch
```

- [ ] Run it; report the attribution table:
      old data/grouping/config -> new data+grouping with old config -> fully new.
- [ ] One sentence per step in the response letter (R3 C13 promised the
      before/after of the regrouping).

## 4. Label-flip accounting (year-matched denominators)

The year-matched population denominators move pairs across the 15% line, and
the corpus itself grew with the recovered storm ids.

- [ ] Count pairs whose category changed between the 2020-fixed and
      year-matched labels; how many unique cyclones they belong to; the new
      positive counts at both thresholds; and, separately, how many records
      the recovered sids added. Goes into Appendix C and Sect. 2.3.

## 5. Confidence intervals before any comparison is called "real"

- [ ] Regenerate the cyclone-clustered bootstrap for every model/strategy
      (`--event-col` now defaults to `sid`):

      ```bash
      python -m src.evaluation.metrics --bootstrap --n-boot 1000
      ```

- [ ] Re-check the manuscript's interval claims against the new CIs — the
      old "detection CIs do not overlap" sentence, and any before/after delta
      quoted in the letter. At the 15% threshold several gaps may not be
      significant.

## 6. Per-threshold framing of model vs baselines

- [ ] From the final run outputs (not figures), state the per-threshold
      result: where the model wins (detection/coverage) and where the
      windspeed-exposure rule wins (severity precision). The hybrid is the
      headline recommendation, not a discussion aside.
- [x] Baselines as deterministic rules (`a352caf`): 33 m/s threshold, no
      fitting, same evaluation path; the historical rule's same-year
      look-ahead is flagged in the docstring — carry that caveat into the
      paper's baseline description.

## 7. Figure and notation consistency

- [ ] The convention is **strictly >15%** everywhere. Stage-2 balancing is
      aligned on this branch; regenerate figure legends from one shared
      constant (the October confusion-matrix figures still mix `>` and `≥`).
- [ ] Walk-forward panels evaluate fewer pairs (earliest cyclones have no
      past data) — state the pair count in the caption.

## 8. Closed since the last version

- [x] **Coast length is geodesic** (`d1dddc2`): WGS84 ellipsoid via
      `pyproj.Geod`, no projection choice at all. Update the Methods sentence
      that described the projected measurement.
- [x] **Rainfall acquisition**: the public pipeline reads NASA PPS granules
      directly with per-product conversions verified against the granule
      metadata; the old OCHA-blob `/10` understatement is gone. The
      "consistently understated, so feature importance holds" argument is
      valid because the old factor was uniform — say exactly that in the
      letter, no more.
- [x] **Storm-id recovery and impact-table correctness** (`649c56f`,
      `213ac78`, `a4342ab`, `def1939`): 13 broken EM-DAT sids resolved and
      verified; impacts attributed only to the DisNo.'s own country;
      `(DisNo., id)` as the table key; target capped and flagged.

## 9. Still open

- [ ] **Selection-filter unification**: `train.py` exempts single-ADM1
      countries from the all-units-affected filter; the search's
      `load_dataset` does not. Decide which rule defines the paper's record
      count and make both match it.
- [ ] **Requirements split**: a minimal `requirements-model.txt`
      (numpy, pandas, scikit-learn, xgboost, joblib, pyarrow, geopandas) for
      the model/search path.
- [ ] **Tests in the release**: the join-test suite written in October sits
      gitignored (`/tests/`). The pure-logic tests (no local data) are worth
      shipping — reviewers read them as rigor (R2 C19).
- [ ] **Threshold sweep** (Appendix B, pair-level counts) and the **TIGGE**
      bookkeeping scripts, regenerated from the final predictions.
- [ ] When all boxes close: tag `v1.0-nhess-r2`, mint the Zenodo DOI, paste
      the prepared sentence into `\codeavailability`.
