# Figures and tables

The notebooks here produce what the manuscript shows. They do not compute
metrics: everything they read comes from `src/evaluation`, so a number in a
figure and the same number in a table come from one computation.

| notebook | produces |
|---|---|
| `metrics_overview.ipynb` | confusion matrices, error violins, strategy and variant comparisons, walk-forward over time |
| `case_studies.ipynb` | the case-study maps: reported impact beside each model's prediction, with the storm track |
| `shap_summary.ipynb` | SHAP beeswarm summaries, one per stage |

`lib.py` holds what the notebooks share — loading a run's output, the impact
bands and their colours, and the plots themselves. It is a module rather than
a notebook so that every figure agrees on what a model output is and how a
prediction is categorised.

## Choosing a run

A run is identified by a suffix on its output folders, taken from
`TC_RUN_SUFFIX`. Unset reads the original run; `_v2` reads the one whose folds
withhold the whole cyclone. Figures follow the same suffix into `figs<suffix>`,
so one run's figures never land on another's.

```bash
TC_RUN_SUFFIX=_v2 jupyter lab          # or set it in the kernel
```

The rule baselines carry no suffix. They have no fitted parameters and are not
re-run with the model, so they keep a single set of outputs.

## Figures are not in the repository

`figs/` and `figs_v2/` are ignored by git. The notebooks rewrite every figure
from the model output on each run, so versioning them would record churn
rather than history. They are kept locally and on the machine the run was
scored on.

## Where the inputs come from

```
src/evaluation/paper_metrics.py     every scored model, with cyclone-clustered intervals
src/evaluation/basin_metrics.py     the same, per cyclone basin
src/evaluation/dataset_counts.py    events, cyclones and administrative units
```

Each writes to `data/output/metrics<suffix>/`. Run them before the notebooks
when the model output has changed.
