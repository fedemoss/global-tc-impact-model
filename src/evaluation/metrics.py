import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score, 
    matthews_corrcoef,
    mean_squared_error,
    cohen_kappa_score
)
import os
from src.config import OUTPUT_DIR
from src.evaluation.bootstrap import bootstrap_binary_metrics, compute_metrics_at_threshold

def compute_binary_metrics(df, target_col="perc_affected_pop_grid_region", pred_col="prediction_perc", thresholds=[0, 15], model_name=None):
    """
    Computes binary classification metrics across specified impact thresholds.
    """
    results = []

    for t in thresholds:
        # The per-threshold maths lives in evaluation.bootstrap so that this
        # table and the bootstrapped one cannot drift apart. nan_on_undefined
        # is False here, which reproduces sklearn's zero_division=0 and the
        # numbers this function returned previously.
        metrics = compute_metrics_at_threshold(
            df, t, target_col=target_col, pred_col=pred_col, nan_on_undefined=False)
        results.append({'model': model_name, 'threshold': t, **metrics})

    return pd.DataFrame(results)

def categorize_values(series):
    """
    Categorize numeric values into 3 ordinal categories:
    0: (-inf, 0]
    1: (0, 15]
    2: (15, inf)
    """
    return pd.cut(series, bins=[-np.inf, 0, 15, np.inf], labels=[0, 1, 2]).astype(int)

def compute_distance_metrics(models_dict, target_col="perc_affected_pop_grid_region", pred_col="prediction_perc"):
    """
    Compute distance/error-based metrics (MAE, MedAE, RMSE, QWK) for multiple models.
    """
    results = []

    for name, df in models_dict.items():
        if 'abs_error' not in df.columns:
            df['abs_error'] = (df[target_col] - df[pred_col]).abs()

        mae = df['abs_error'].mean()
        medae = df['abs_error'].median()
        rmse = np.sqrt(mean_squared_error(df[target_col], df[pred_col]))
        
        # Categorize for QWK
        reported_cat = categorize_values(df[target_col])
        predicted_cat = categorize_values(df[pred_col])
        qwk = cohen_kappa_score(reported_cat, predicted_cat, weights='quadratic')
        
        results.append({
            'model': name,
            'MAE': mae,
            'MedAE': medae,
            'RMSE': rmse,
            'QWK': qwk
        })
    
    return pd.DataFrame(results)

def load_and_evaluate_all_models(level="adm1", strategy="global",
                                 bootstrap=False, n_boot=1000, event_col="DisNo.",
                                 random_state=42):
    """
    Loads model outputs from OUTPUT_DIR and computes standard tables.

    With `bootstrap=True` the binary table also carries event-level confidence
    intervals (`<metric>_ci_low` / `_ci_high`) and, per metric, the number of
    replicates where it was undefined. Resampling is over `event_col`, never
    over rows - see evaluation/bootstrap.py for why that distinction decides
    whether the intervals mean anything.
    """
    model_names = [
        "historical", 
        "windspeed-exposed", 
        "windspeed-historical", 
        "2-stage-XGBoost"
    ]
    
    models_dict = {}
    binary_results = []
    
    print(f"Evaluating {level.upper()} level results using {strategy} strategy...")
    
    for model_name in model_names:
        folder_path = OUTPUT_DIR / f"model_output/{model_name}_{strategy}_{level}"
        
        if not folder_path.exists():
            print(f"Warning: Results folder not found for {model_name}")
            continue
            
        # Combine all per-event CSVs into one evaluation dataframe
        csv_files = list(folder_path.glob("predictions_event_*.csv"))
        if not csv_files:
            continue
            
        df_model = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        models_dict[model_name] = df_model
        
        # Compute binary metrics, with event-level CIs when asked for
        if bootstrap:
            if event_col not in df_model.columns:
                print(f"Warning: {model_name} has no {event_col!r} column; "
                      f"falling back to point estimates without CIs")
                metrics_bin = compute_binary_metrics(df_model, model_name=model_name)
            else:
                metrics_bin = bootstrap_binary_metrics(
                    df_model, event_col=event_col, model_name=model_name,
                    n_boot=n_boot, random_state=random_state)
        else:
            metrics_bin = compute_binary_metrics(df_model, model_name=model_name)
        binary_results.append(metrics_bin)
        
    if not models_dict:
        print("No evaluation data found. Run train.py first.")
        return None, None
        
    all_binary_metrics = pd.concat(binary_results, ignore_index=True)
    all_binary_metrics = all_binary_metrics.sort_values(by=['threshold', 'model']).reset_index(drop=True)
    
    distance_metrics = compute_distance_metrics(models_dict)
    
    return all_binary_metrics, distance_metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument("--level", default="adm1")
    parser.add_argument("--strategy", default="global")
    parser.add_argument("--bootstrap", action="store_true",
                        help="add event-level bootstrap confidence intervals")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--event-col", default="DisNo.",
                        help="unit resampled by the bootstrap (a cyclone/event, not a row)")
    args = parser.parse_args()

    bin_metrics, dist_metrics = load_and_evaluate_all_models(
        level=args.level, strategy=args.strategy,
        bootstrap=args.bootstrap, n_boot=args.n_boot, event_col=args.event_col)

    if bin_metrics is not None:
        print("\n--- Binary Metrics ---")
        if args.bootstrap:
            from src.evaluation.bootstrap import format_ci
            compact = bin_metrics[["model", "threshold", "n_events"]].copy()
            for metric in ("precision", "recall", "f1", "csi"):
                compact[metric] = bin_metrics.apply(lambda r: format_ci(r, metric), axis=1)
            print(compact.to_string(index=False))
            print("\n(values are point estimate [95% CI], resampled over events)")
        else:
            print(bin_metrics.to_string())

        print("\n--- Distance Metrics ---")
        print(dist_metrics.to_string())