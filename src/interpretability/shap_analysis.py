import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from src.config import OUTPUT_DIR, FEATURES

def load_trained_model(model_path):
    """Loads the serialized 2-stage model components."""
    # Assuming the 2-stage model saves its components as .joblib or .pkl
    classifier = joblib.load(model_path / "classifier.joblib")
    regressor = joblib.load(model_path / "regressor.joblib")
    return classifier, regressor

def run_shap_explanation(df, model, feature_names, output_path, stage_name="regressor"):
    """
    Computes SHAP values and saves the summary plot.
    
    Args:
        df: The validation/test dataframe containing features.
        model: The fitted XGBoost model component.
        feature_names: List of strings used for training.
        output_path: Directory to save plots.
        stage_name: 'classifier' or 'regressor' for naming files.
    """
    X = df[feature_names]
    
    # Initialize TreeExplainer (specifically for XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 1. Summary Plot (The 'Beeswarm' plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Summary: {stage_name.capitalize()} Stage")
    plt.tight_layout()
    plt.savefig(output_path / f"shap_summary_{stage_name}.png", dpi=300)
    plt.close()

    # 2. Save SHAP values for further analysis
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(output_path / f"shap_values_{stage_name}.csv", index=False)
    
    return shap_values

def generate_dependence_plots(shap_values, X, feature_names, output_path):
    """Generates dependence plots for the top 3 most important features."""
    # Get mean absolute SHAP values to find top features
    top_inds = np.argsort(np.abs(shap_values).mean(0))[-3:]
    top_features = [feature_names[i] for i in top_inds]

    for feat in top_features:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X, show=False)
        plt.title(f"SHAP Dependence: {feat}")
        plt.tight_layout()
        plt.savefig(output_path / f"shap_dependence_{feat}.png", dpi=300)
        plt.close()

def main_shap_analysis(model_run_name="2-stage-XGBoost_global_grid"):
    """
    Main entry point for interpretability analysis.
    Uses the output from a specific training run.
    """
    run_dir = OUTPUT_DIR / "model_output" / model_run_name
    shap_out_dir = OUTPUT_DIR / "interpretability" / model_run_name
    shap_out_dir.mkdir(parents=True, exist_ok=True)

    # Load the model and a sample of the data used for that run
    # Note: For SHAP, we usually use the validation/test set or a representative sample
    try:
        classifier, regressor = load_trained_model(run_dir)
        # Load the predictions file to get the feature data associated with those events
        df_results = pd.read_csv(run_dir / "all_predictions_compiled.csv")
    except FileNotFoundError:
        print(f"Required model files or results not found in {run_dir}")
        return

    print("Generating SHAP explanations for the Regressor stage...")
    # Typically, the regressor stage provides more insight into 'impact magnitude'
    shap_vals_reg = run_shap_explanation(
        df_results, 
        regressor, 
        FEATURES, 
        shap_out_dir, 
        stage_name="regressor"
    )
    
    generate_dependence_plots(shap_vals_reg, df_results[FEATURES], FEATURES, shap_out_dir)
    
    print(f"Interpretability plots saved to {shap_out_dir}")

if __name__ == "__main__":
    main_shap_analysis()