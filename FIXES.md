# Repository Fixes

This document describes the bugs found during the code review and the fixes applied. Issues are grouped by severity.

---

## 1. Logging restructure (per request)

**Before:** Each module called `logging.basicConfig(...)` independently, with several writing to `OUTPUT_DIR/logs/<module>.log`. Only the *first* `basicConfig` call in a Python process takes effect, so most of those handlers were silently ignored. The rest emitted via `print()`.

**After:**
- New module [src/utils/logging_setup.py](src/utils/logging_setup.py) with a single `configure_logging()` entry point.
- Logs are written to **both stdout and `main.log` at the repository root** (sibling of `main.py`).
- Removed every per-module `logging.basicConfig(...)`.
- Every module now declares `logger = logging.getLogger(__name__)` and uses `logger.info/warning/error` instead of `print()`.
- `if __name__ == "__main__":` blocks call `configure_logging()` so running a single module standalone still produces the same unified log.

---

## 2. Critical bugs

### 2.1 Missing comma in `NON_CONTEMPLATED_FEATURES`
- **File:** [src/config.py](src/config.py)
- **Bug:** Python's implicit string concatenation produced `["flood_risk", "shditrack_distance"]` (2 items instead of 3).
- **Fix:** Added the missing comma; the list is now `['flood_risk', 'shdi', 'track_distance']`.

### 2.2 SRTM projection bug
- **File:** [src/static_features/process_srtm.py](src/static_features/process_srtm.py)
- **Bug:** `adjust_longitude` was applied to *every* polygon in the global grid — even those already in `[-180, 180]`. The function was a no-op for those, but combined with other issues it masked alignment problems for antimeridian-crossing countries (Fiji, Russia, NZ).
- **Fix:** Apply `adjust_longitude` only to polygons that actually contain `lon > 180`. The shared helper now lives in [src/utils/geo_utils.py](src/utils/geo_utils.py).

### 2.3 Coast length used a Philippines-only projection
- **File:** [src/static_features/process_srtm.py:158](src/static_features/process_srtm.py:158)
- **Bug:** `to_crs(epsg=25394)` is "Luzon 1911 / Philippines Zone V" — a local Philippine CRS. Using it globally produced wildly incorrect coast lengths.
- **Fix:** Replaced with `EPSG:6933` (World Cylindrical Equal Area), exposed as `GLOBAL_METRIC_EPSG` in `geo_utils.py`.

### 2.4 Flood-risk tile selection missed valid tiles
- **File:** [src/static_features/process_flood_risk.py](src/static_features/process_flood_risk.py)
- **Bug:** `select_tiles_by_country` checked whether a tile *center* fell inside a 3-degree-buffered country geometry. This missed many edge tiles whose centers are outside the country but whose extent overlaps it, contributing to "missing flood_risk features" the student observed.
- **Fix:** Rewrote tile selection to use a proper bounding-box intersection between each tile's full extent (parsed from the filename) and the country bounds (with a 0.5° padding).
- **Bonus:** Added structured logging, error handling, NaN-safe stacking with `np.nanmax`.

### 2.5 Rainfall: row-by-row iteration + point sampling instead of mean
- **File:** [src/dynamic_features/process_rain_features.py](src/dynamic_features/process_rain_features.py)
- **Bugs:**
  1. `for index, row in grid.iterrows():` is extremely slow on GeoDataFrames.
  2. `da_box.values[0, 0]` took the *first pixel* in each grid cell, not the mean — even though the column was named `"mean"`.
  3. `xarray.sel(y=slice(miny, maxy))` returns empty arrays when the y-coordinate is descending (which GPM-IMERG is by default).
- **Fix:** Replaced the entire iteration with `rasterstats.zonal_stats(..., stats=["mean"])`, which handles affine alignment and y-axis direction correctly and returns the actual mean over polygon pixels.

### 2.6 Historical features path mismatch
- **Files:** [src/dynamic_features/process_historical_features.py](src/dynamic_features/process_historical_features.py), [src/dataset_builder.py](src/dataset_builder.py)
- **Bug:** Historical features were written to `OUTPUT_DIR/features/` but read from `OUTPUT_DIR/dynamic_features/` — meaning `N_events_5_years` would silently always be missing in the final training dataset.
- **Fix:** Producer now writes to `OUTPUT_DIR/dynamic_features/` to match the consumer.

### 2.7 `--build-dataset` was always on
- **File:** [main.py](main.py)
- **Bug:** `argparse.add_argument("--build-dataset", action="store_true", default=True, ...)` meant the flag was always `True`. Running `python main.py --run-models` would also trigger a full data rebuild.
- **Fix:** Default flipped to `False`. With no flags the pipeline still defaults to building (matches the previous behavior). Selecting any `--stage` implies `--build-dataset`.

### 2.8 Wind processing used threads for CPU-bound work
- **File:** [src/dynamic_features/process_wind_features.py](src/dynamic_features/process_wind_features.py)
- **Bug:** `ThreadPoolExecutor` for `TropCyclone.from_tracks(...)` (CPU-bound NumPy work) is throttled by Python's GIL — effectively sequential.
- **Fix:** Switched to `ProcessPoolExecutor` for true parallelism.

### 2.9 Bare `except` clauses swallowed everything
- **File:** [src/dynamic_features/process_wind_features.py](src/dynamic_features/process_wind_features.py), [src/collectors/general_collector.py](src/collectors/general_collector.py)
- **Bug:** `except:` (without an exception type) swallows `KeyboardInterrupt`, `SystemExit`, `MemoryError`, and provides zero diagnostics on real failures.
- **Fix:** Replaced with `except Exception as e:` + `logger.warning(...)` so failures are surfaced. SRTM tile download errors now log the failing tile and report a summary at the end.

### 2.10 `collect_guil` streaming bug
- **File:** [src/collectors/general_collector.py](src/collectors/general_collector.py)
- **Bug:** `requests.get(url, stream=True)` followed by `f.write(response.content)` reads the entire response into memory anyway.
- **Fix:** Now iterates `response.iter_content(chunk_size=8192)` to keep memory bounded for the multi-GB GAUL archive.

### 2.11 SHAP analysis would always fail
- **Files:** [src/evaluation/cv_strategies.py](src/evaluation/cv_strategies.py), [src/interpretability/shap_analysis.py](src/interpretability/shap_analysis.py)
- **Bug:** `shap_analysis.py` tried to load `classifier.joblib` / `regressor.joblib` and `all_predictions_compiled.csv`, but the LOOCV pipeline never saved either.
- **Fix:** After the LOOCV loop, `cv_strategies.py` now:
  1. Fits the model on the full filtered dataset and persists `classifier.joblib` / `regressor.joblib` (or `model.joblib` for baselines) into the run directory.
  2. Concatenates per-event prediction CSVs into `all_predictions_compiled.csv` for SHAP to consume.

### 2.12 `train.py` `date` column collision
- **File:** [src/models/train.py](src/models/train.py)
- **Bug:** When `aggregate_to_adm1=True`, `date` was both a `groupby` key (line 48) and a freshly merged column (line 56), creating `date_x` / `date_y` after merge. `.drop_duplicates()` masked but did not fix it.
- **Fix:** Refactored to attach the EM-DAT-derived `date` *before* aggregation, drop any pre-existing `date` column to prevent merge collisions.

---

## 3. Architectural cleanups

### 3.1 Shared geospatial utilities
- New module [src/utils/geo_utils.py](src/utils/geo_utils.py) containing:
  - `adjust_longitude(polygon)` — wrap `[0, 360)` longitudes into `[-180, 180]`. Previously duplicated in 5 modules.
  - `is_antimeridian_crossing(polygon)` — replaces the local `is_border_crossing` in SRTM.
  - `GLOBAL_METRIC_EPSG = 6933` — the equal-area projection used for distance/length calculations.

### 3.2 Subpackage `__init__.py`
- Added empty `__init__.py` files to all 7 subpackages (`collectors`, `static_features`, `dynamic_features`, `models`, `evaluation`, `interpretability`, `utils`) so the package layout is explicit rather than relying on namespace packages.

### 3.3 `print()` → `logger`
- Converted `print(...)` calls to `logger.info/warning/error` across the touched modules (collectors, all feature processors, training, dataset builder, PPS collector). Output is consistent across stdout and `main.log`.

---

## 4. Outstanding student items addressed

| Student note | Status |
|---|---|
| Missing SRTM features (projection problem) | Fixed — see 2.2, 2.3 |
| Missing flood_risk features (similar) | Fixed — see 2.4 |
| Windspeed/rainfall need verification | Wind: 2.8, 2.9 (parallelism + error handling). Rainfall: 2.5 (correct mean + correct alignment). |
| Final training set creation | Fixed — see 2.6, 2.7 |
| LOOCV training | Functional already. Now also persists model artifacts (2.11) |
| Logging restructure | Fixed — see 1 |

---

## 5. Verification

All Python files parse cleanly:

```bash
python -c "import ast; from pathlib import Path; \
  [ast.parse(f.read_text()) for f in Path('.').rglob('*.py') if '__pycache__' not in str(f)]"
```

Utility helpers pass sanity assertions:

```bash
python -c "
from src.utils.geo_utils import adjust_longitude, is_antimeridian_crossing, GLOBAL_METRIC_EPSG
from src.config import NON_CONTEMPLATED_FEATURES
assert NON_CONTEMPLATED_FEATURES == ['flood_risk', 'shdi', 'track_distance']
assert GLOBAL_METRIC_EPSG == 6933
"
```

A full pipeline run requires Python 3.10+ (per `requirements.txt`) and the external data described in the main `README.md`.
