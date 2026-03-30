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

def compute_binary_metrics(df, target_col="perc_affected_pop_grid_region", pred_col="prediction_perc", thresholds=[0, 15], model_name=None):
    """
    Computes binary classification metrics across specified impact thresholds.
    """
    results = []
    
    # Ensure absolute error exists
    if 'abs_error' not in df.columns:
        df['abs_error'] = (df[target_col] - df[pred_col]).abs()
    
    for t in thresholds:
        # Binarize
        df['reported_bin'] = (df[target_col] > t).astype(int)
        df['predicted_bin'] = (df[pred_col] > t).astype(int)

        # Confusion matrix components
        TP = ((df['predicted_bin'] == 1) & (df['reported_bin'] == 1)).sum()
        TN = ((df['predicted_bin'] == 0) & (df['reported_bin'] == 0)).sum()
        FP = ((df['predicted_bin'] == 1) & (df['reported_bin'] == 0)).sum()
        FN = ((df['predicted_bin'] == 0) & (df['reported_bin'] == 1)).sum()

        # Binary metrics
        precision = precision_score(df['reported_bin'], df['predicted_bin'], zero_division=0)
        recall = recall_score(df['reported_bin'], df['predicted_bin'], zero_division=0)
        f1 = f1_score(df['reported_bin'], df['predicted_bin'], zero_division=0)
        accuracy = accuracy_score(df['reported_bin'], df['predicted_bin'])
        
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        csi = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

        mean_abs_error = df['abs_error'].mean()
        median_abs_error = df['abs_error'].median()

        results.append({
            'model': model_name,
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'fpr': fpr,
            'fnr': fnr,
            'f1': f1,
            'csi': csi,
            'mean_abs_error': mean_abs_error,
            'median_abs_error': median_abs_error
        })
    
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

def load_and_evaluate_all_models(level="adm1", strategy="global"):
    """
    Loads model outputs from OUTPUT_DIR and computes standard tables.
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
        
        # Compute binary metrics
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
    # Example execution: Compute metrics for ADM1 Global LOOCV
    bin_metrics, dist_metrics = load_and_evaluate_all_models(level="adm1", strategy="global")
    
    if bin_metrics is not None:
        print("\n--- Binary Metrics ---")
        print(bin_metrics.to_string())
        
        print("\n--- Distance Metrics ---")
        print(dist_metrics.to_string())