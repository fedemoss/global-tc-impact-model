import argparse
import logging
import sys
from src.config import INPUT_DIR

# Collectors
from src.collectors.general_collector import download_all_public_data

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
from src.dynamic_features.process_historical_features import generate_all_historical_features

# Build & Model
from src.dataset_builder import compile_global_dataset
from src.models.train import execute_training_run
from src.interpretability.shap_analysis import main_shap_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    parser = argparse.ArgumentParser(description="Global TC Impact Data Factory")
    
    # Operation Mode
    parser.add_argument("--build-dataset", action="store_true", default=True,
                        help="Generate the full training_dataset.parquet")
    parser.add_argument("--run-models", action="store_true", 
                        help="Train the 2-stage XGBoost model")
    parser.add_argument("--run-interpretability", action="store_true",
                        help="Run SHAP analysis on trained models")

    # Selective Stage (for debugging)
    parser.add_argument("--stage", type=str, 
                        choices=["collect", "grid", "static", "dynamic", "build"])

    args = parser.parse_args()

    # --- DATA PIPELINE ---
    if args.build_dataset:
        
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
            generate_all_rain_features()
            generate_all_historical_features()

        if not args.stage or args.stage == "build":
            logging.info("Stage 5: Compiling Final Training Dataset...")
            compile_global_dataset()

    # --- OPTIONAL SCIENCE ---
    if args.run_models:
        logging.info("Science: Running 2-Stage XGBoost Model...")
        execute_training_run("2-stage-XGBoost", strategy="global", aggregate_to_adm1=True)

    if args.run_interpretability:
        logging.info("Science: Running SHAP Analysis...")
        main_shap_analysis()

if __name__ == "__main__":
    main()