# Rerun checklist 


---

## 1. Provenance of the rerun (confirm)

- [ ] **Folds = physical cyclones.** The LOOCV log prints
      `N physical cyclones / M country-event records / rows -> K folds`;
      confirm K equals the cyclone count, not the record count.
- [ ] **Which commit ran.** Record the exact commit hash of the rerun in the
      run directory (e.g. `git rev-parse HEAD > <run_dir>/COMMIT`). If it
      predates the sid-default commit on this branch, state explicitly that
      `event_col="sid"` was passed.
- [ ] **Which hyperparameters ran.** If `data/model_hyperparameters.json`
      came from the *old* (DisNo.-grouped) search, re-run the search first —
      the 0.6 threshold was selected under the old grouping:

      ```bash
      python test/hyperparameter_search.py --n-iter 80 --n-events 200
      ```

      The written JSON records `group_col` and the exact search/hold-out sid
      lists; keep it with the run.
- [ ] **Which specification ran** — with or without `N_events_5_years`?
      State it; the with/without pair is the ablation (Appendix E), and the
      primary-specification decision is still open with KK.

## 2. Attribution runs 

"How much of 0.42 ->  0.34 is leakage removed vs model re-tuned vs data
corrected?" One extra run splits the two biggest factors — the old fixed
configuration under the new grouping and data:

```python
from src.models.train import execute_training_run
# corrected evaluation, round-1 model configuration:
execute_training_run("2-stage-XGBoost", "global", aggregate_to_adm1=True,
                     use_tuned=False)   # hook added on this branch
```

- [ ] Run it; report the three-row attribution table:
      old data/grouping/config -> new data+grouping with old config -> fully new.
- [ ] One sentence per step in the response letter (R3 C13 promised the
      before/after of the regrouping).

## 3. Label-flip accounting (618 -> 647 high pairs)

The year-matched population denominators move pairs across the 15% line
(+29 net). Reviewers must see this is the denominator fix, not a bug.

- [ ] Count pairs whose category changed between the 2020-fixed and
      year-matched labels; how many unique cyclones they belong to; the new
      positive counts at both thresholds. Goes into Appendix C.

## 4. Confidence intervals before any comparison is called "real"

- [ ] Regenerate the cyclone-clustered bootstrap for every model/strategy:

      ```bash
      python -m src.evaluation.metrics --bootstrap --n-boot 1000
      ```

- [ ] Re-check the manuscript's interval claims against the new CIs —
      in particular the old "detection CIs do not overlap" sentence, and any
      before/after delta quoted in the letter. At the 15% threshold several
      gaps may not be significant.

## 5. The baseline inversion 

From the common-set matrices: at the **15% threshold** windspeed-exposed now
has precision ≈ 0.23 / F1 ≈ 0.30 against the model's 0.08 / 0.15, while the
model clearly wins **detection** (F1 ≈ 0.34 vs 0.25). 

- [ ] Verify these from the run outputs (not from the figure) and put the
      per-threshold framing in the group email and the manuscript: model for
      detection, exposure rule for severity — the hybrid becomes the headline
      recommendation, not a discussion aside.

## 6. Figure and notation consistency

- [ ] The two confusion-matrix figures disagree on the boundary
      (`> 15%` vs `≥ 15%`). The convention is **strictly >** everywhere
      (stage-2 balancing aligned to `> 15` on this branch); regenerate the
      figure legends from one shared constant.
- [ ] Walk-forward panels evaluate fewer pairs (earliest cyclones have no
      past data) — state the pair count in the caption so the panels aren't
      read as the same denominator.

## 7. Coast length — decide before the feature freeze

`coast_length_meters` is currently computed in EPSG:25394 (Luzon 1911 /
Philippines zone V). The note in `process_srtm.py` calling EPSG:6933 a "local
UTM zone" is wrong — 6933 is World Cylindrical Equal Area — but using it for
*lengths* was also wrong (equal-area =/= equidistant). Neither CRS is the right
tool; the clean fix needs no CRS at all:

```python
from pyproj import Geod
GEOD = Geod(ellps="WGS84")
grid_line["coast_length_meters"] = grid_line.geometry.apply(GEOD.geometry_length)
```

- [ ] Decide with KK: adopt geodesic length (changes a model feature -> must
      land before, not after, the frozen rerun) or document the 25394 choice
      honestly as a legacy approximation.

## 8. Still open

- [ ] **Selection-filter unification**: `src/models/train.py` exempts
      single-ADM1 countries from the all-units-affected filter;
      `test/hyperparameter_search.py:load_dataset` does not. The search pool
      and the LOOCV pool therefore differ slightly. Decide which rule
      produced the paper's 780 records and make both match it.
- [ ] **Requirements split**: a minimal `requirements-model.txt`
      (numpy, pandas, scikit-learn, xgboost, joblib, pyarrow, geopandas) for
      the model/search path
- [ ] **Threshold sweep** (Appendix B, pair-level counts) and the **TIGGE**
      bookkeeping scripts, regenerated from the rerun predictions.
- [ ] When all boxes close: tag `v1.0-nhess-r2`, mint the Zenodo DOI, paste
      the prepared sentence into `\codeavailability`.
