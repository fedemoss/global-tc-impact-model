import argparse
import logging
import sys
from src.config import INPUT_DIR, IMERG_PRODUCT, IMERG_PRODUCTS

# Collectors
from src.collectors.general_collector import download_all_public_data
from src.collectors.forecast_collector import (DEFAULT_PROVIDER,
                                               PROVIDERS as FORECAST_PROVIDERS)

# Grid
from src.static_features.grid_cells import main_grid_generation

# Static Processors
from src.utils.region_matching import create_region_dataset
from src.static_features.process_worldpop import process_all_worldpop
from src.static_features.process_srtm import process_all_srtm
from src.static_features.process_jrc import process_all_jrc
from src.static_features.process_landslide import process_all_landslide
from src.static_features.process_storm_surges import process_all_surges
from src.static_features.process_flood_risk import process_all_flood
from src.static_features.process_shdi import process_all_shdi
from src.static_features.process_gadm import process_gadm_adm2

# Dynamic Processors
from src.dynamic_features.process_emdat import process_emdat_events
from src.dynamic_features.process_wind_features import generate_all_wind_features
from src.dynamic_features.process_rain_features import generate_all_rain_features
from src.dynamic_features.process_historical_features import (DEFAULT_LEVEL, LEVELS,
                                                              generate_all_historical_features)

# Build & Model
from src.dataset_builder import compile_global_dataset
from src.models.baselines import build_all_baselines
from src.models.train import execute_training_run
from src.interpretability.shap_analysis import main_shap_analysis
from src.utils.logging_setup import configure_logging



def resolve_features(args):
    """The feature set to fit on, and the name that keeps its output apart.

    Returns (None, None) for the full set, so an ordinary run is unaffected.
    """
    from src.config import FEATURES

    if args.features and args.exclude_features:
        raise SystemExit("--features and --exclude-features are mutually exclusive")

    if args.features:
        unknown = sorted(set(args.features) - set(FEATURES))
        if unknown:
            raise SystemExit(f"not features of this model: {unknown}\n"
                             f"available: {sorted(FEATURES)}")
        chosen = [f for f in FEATURES if f in set(args.features)]
        name = args.run_name or "subset_" + "_".join(sorted(args.features))[:60]
        return chosen, name

    if args.exclude_features:
        unknown = sorted(set(args.exclude_features) - set(FEATURES))
        if unknown:
            raise SystemExit(f"not features of this model: {unknown}\n"
                             f"available: {sorted(FEATURES)}")
        chosen = [f for f in FEATURES if f not in set(args.exclude_features)]
        if not chosen:
            raise SystemExit("every feature was excluded")
        name = args.run_name or "no_" + "_".join(sorted(args.exclude_features))
        return chosen, name

    return None, args.run_name


def main():
    parser = argparse.ArgumentParser(description="Global TC Impact Data Factory")

    # Operation Mode
    parser.add_argument("--build-dataset", action="store_true",
                        help="Generate the full training_dataset.parquet (implied when no "
                             "--run-* flag is given, and by --stage)")
    parser.add_argument("--run-models", action="store_true", 
                        help="Train the 2-stage XGBoost model")
    parser.add_argument("--run-interpretability", action="store_true",
                        help="Run SHAP analysis on trained models")
    parser.add_argument("--models", nargs="+", default=["2-stage-XGBoost"],
                        choices=["2-stage-XGBoost", "2-stage-GLM"],
                        help="Fitted models to run with --run-models. '2-stage-GLM' "
                             "is the statistical baseline (logistic occurrence, "
                             "fractional-logit severity) on the same predictors, "
                             "folds and aggregation; tune it first with "
                             "src/models/glm_search.py")
    parser.add_argument("--with-baselines", action="store_true",
                        help="Also run the three reference baselines, so the "
                             "evaluation can compare against them")
    parser.add_argument("--exclude-features", nargs="+", default=None,
                        metavar="FEATURE",
                        help="Fit without these features, for an ablation. The "
                             "run is written to its own folder, named after "
                             "what was left out unless --run-name says "
                             "otherwise")
    parser.add_argument("--features", nargs="+", default=None, metavar="FEATURE",
                        help="Fit on exactly these features. Mutually "
                             "exclusive with --exclude-features")
    parser.add_argument("--run-name", default=None,
                        help="Suffix for the output folder, so that runs "
                             "differing only in their features stay apart")
    parser.add_argument("--train-at", choices=["grid", "adm1"], default="grid",
                        help="Level the model is fitted at. 'grid' fits per "
                             "cell and aggregates the predictions to ADM1 "
                             "afterwards; 'adm1' fits the administrative units "
                             "directly. Results are kept in separate folders")
    parser.add_argument("--strategies", nargs="+", default=["global"],
                        choices=["global", "walk_forward", "geo_constrained"],
                        help="Cross-validation strategies to run with --run-models. "
                             "'global' trains on every other cyclone, 'walk_forward' "
                             "only on cyclones preceding the test one, and "
                             "'geo_constrained' only on cyclones in the same basin. "
                             "Each writes to its own output folder")
    parser.add_argument("--collect-forecasts", action="store_true",
                        help="Download archived ensemble cyclone-track "
                             "forecasts for the events the model is evaluated "
                             "on. Resumable: cycles already on disk are kept")
    parser.add_argument("--run-forecast-study", action="store_true",
                        help="Predict impact from each archived forecast "
                             "cycle, leaving the storm out of training")
    parser.add_argument("--forecast-provider", default=DEFAULT_PROVIDER,
                        choices=sorted(FORECAST_PROVIDERS),
                        help="Which centre's ensemble to use. Coverage of the "
                             "event set differs widely between them, and only "
                             "ECMWF reaches before 2008")
    parser.add_argument("--forecast-lead-hours", type=float, default=120.0,
                        help="How far into each forecast to follow a member, "
                             "in hours")
    parser.add_argument("--predict-unit", choices=["DisNo.", "sid"], default="DisNo.",
                        help="Unit predicted per cross-validation fold: the "
                             "country-event record impact is reported at "
                             "(default), or the physical cyclone")
    parser.add_argument("--withhold-unit", choices=["DisNo.", "sid"], default="sid",
                        help="Unit withheld from each fold's training set. The "
                             "default withholds the whole cyclone, so the other "
                             "countries it struck cannot train the fold "
                             "predicting this one. Set it equal to "
                             "--predict-unit to withhold only the record under "
                             "test. Non-default pairs write to their own folder")
    parser.add_argument("--recompute", action="store_true",
                        help="Overwrite existing per-event predictions instead of "
                             "reusing them. Required after changing the model or its "
                             "hyperparameters, otherwise every event is skipped and "
                             "the previous predictions are kept")

    # Selective Stage (for debugging)
    parser.add_argument("--stage", type=str,
                        choices=["collect", "grid", "static", "dynamic", "build"])

    # Rainfall source. Both routes produce the same `rainfall_max_24h` feature
    # and agree to correlation 0.9994; they differ in precision and download
    # size. See IMERG_PRODUCT in src/config.py.
    parser.add_argument("--imerg-product", type=str, default=IMERG_PRODUCT,
                        choices=list(IMERG_PRODUCTS),
                        help=("IMERG product for rainfall. 'half_hourly' (default) sums the "
                              "48 half-hourly granules per day: 0.1 mm resolution, ~166 MB "
                              "per storm. 'daily' uses one granule per date and converts its "
                              "rate to mm/day: 2.4 mm/day resolution, ~1 MB per date, and "
                              "dates are shared between storms."))
    parser.add_argument("--historical-level", type=str, default=DEFAULT_LEVEL,
                        choices=list(LEVELS),
                        help=("What N_events_5_years counts a past cyclone against. "
                              "'grid' (default) counts, per cell, the past events "
                              "whose reported impacting area covered it, so the "
                              "feature varies within a country. 'country' counts "
                              "them anywhere in the country and gives that number "
                              "to every cell."))

    args = parser.parse_args()

    # Without this the root logger sits at WARNING and the entire pipeline runs
    # silent -- every logger.info below and in every module goes nowhere.
    configure_logging()

    # Build the dataset when asked for explicitly (--build-dataset or --stage),
    # or when nothing else was requested -- so a bare `python main.py` is still
    # the full factory, but `--run-models` alone no longer downloads the world.
    build_dataset = args.build_dataset or args.stage is not None or not (
        args.run_models or args.run_interpretability
        or args.collect_forecasts or args.run_forecast_study
    )

    # --- DATA PIPELINE ---
    if build_dataset:
        
        if not args.stage or args.stage == "collect":
            logging.info("Stage 1: Downloading Public Data...")
            download_all_public_data()

        if not args.stage or args.stage == "grid":
            logging.info("Stage 2: Initializing 0.1° Global Grid...")
            main_grid_generation()

        if not args.stage or args.stage == "static":
            logging.info("Stage 3: Processing Static Spatial Layers...")
            process_gadm_adm2()
            create_region_dataset() # Create also a dataset with region information
            process_all_worldpop()
            process_all_srtm()
            process_all_jrc()
            process_all_landslide()
            process_all_surges()
            process_all_flood()
            process_all_shdi()

        if not args.stage or args.stage == "dynamic":
            logging.info("Stage 4: Processing Dynamic Hazard & Impact Layers...")
            # Anchor the dynamic processing to EM-DAT targets first
            process_emdat_events() 
            # Process hazards
            generate_all_wind_features()
            generate_all_rain_features(product=args.imerg_product)
            generate_all_historical_features(level=args.historical_level)

        if not args.stage or args.stage == "build":
            logging.info("Stage 5: Compiling Final Training Dataset...")
            compile_global_dataset()

    # --- OPTIONAL SCIENCE ---
    if args.run_models:
        # Fitted on grid cells, matching the hyperparameter search; predictions
        # are aggregated to ADM1 afterwards, which is the level EM-DAT reports
        # at and the level the baselines are compared on.
        features, run_name = resolve_features(args)
        for model_name in args.models:
            for strategy in args.strategies:
                logging.info(f"Science: Running {model_name} ({strategy}, "
                             f"fitted at {args.train_at})...")
                execute_training_run(model_name, strategy=strategy,
                                     aggregate_to_adm1=(args.train_at == "adm1"),
                                     event_col=args.predict_unit,
                                     exclude_col=args.withhold_unit,
                                     recompute=args.recompute,
                                     features=features, run_name=run_name)

        if args.with_baselines:
            # The baselines are deterministic rules with no fitted parameter,
            # so they are evaluated in one pass over the dataset rather than
            # through the fold loop, and their predictions are kept whole
            # rather than split per event.
            logging.info("Science: Scoring the reference baselines...")
            build_all_baselines()

    if args.collect_forecasts or args.run_forecast_study:
        from src.evaluation.forecast_study import (collect_study_forecasts,
                                                   run_forecast_study)
        events = None
        if args.collect_forecasts:
            logging.info(f"Forecasts: collecting from "
                         f"{args.forecast_provider}...")
            events = collect_study_forecasts(provider=args.forecast_provider)
        if args.run_forecast_study:
            logging.info("Forecasts: predicting impact from each cycle...")
            run_forecast_study(provider=args.forecast_provider,
                               lead_hours=args.forecast_lead_hours,
                               events=events)

    if args.run_interpretability:
        logging.info("Science: Running SHAP Analysis...")
        main_shap_analysis()

if __name__ == "__main__":
    main()