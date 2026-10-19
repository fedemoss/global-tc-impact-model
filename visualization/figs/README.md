# Figures

Written by `metrics_overview.ipynb`, which regenerates all of them on each
run. They are tracked so the current results can be read without the model
outputs, which are far too large to carry.

| file | what it shows |
| --- | --- |
| `categorical_confusion_matrices.png` | the model against the three reference baselines, by impact category |
| `absolute_error_violins.png` | absolute error per model, split by the reported band |
| `strategy_confusion_matrices.png` | the model under each cross-validation strategy, by category |
| `f1_by_strategy.png` | F1 per strategy at both thresholds, with cyclone-clustered intervals |

Colour in the confusion matrices is normalised within each row, with a
separate scale per row: the no-impact row is two orders of magnitude larger
than the others and would otherwise flatten them.
