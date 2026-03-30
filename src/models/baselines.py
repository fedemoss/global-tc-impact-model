import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# ---------------------------------------------------------
# 1. Historical Model (Median-based)
# ---------------------------------------------------------
class HistoricalModel:
    """
    'historical' baseline: predicts the median of past impact data based on the region.
    Does not use machine learning; relies purely on historical statistical medians.
    """
    def __init__(self, region_col="iso3", target_name="perc_affected_pop_grid_region"):
        self.region_col = region_col
        self.target_name = target_name
        self.regional_medians = None
        self.global_median = 0
        
    def fit(self, X, y=None):
        target = y if y is not None else X[self.target_name]
        df = X.copy()
        df[self.target_name] = target
        
        # Calculate median past impact per region
        self.regional_medians = df.groupby(self.region_col)[self.target_name].median().reset_index()
        self.regional_medians.rename(columns={self.target_name: "regional_median_pred"}, inplace=True)
        
        # Fallback global median for completely unseen regions
        self.global_median = df[self.target_name].median()
        return self
        
    def predict(self, X):
        df = X.copy()
        df = df.merge(self.regional_medians, on=self.region_col, how="left")
        
        # Fill regions with no past data with the global median
        preds = df["regional_median_pred"].fillna(self.global_median).values
        return preds
        
    def train_and_predict(self, df_train, df_test):
        self.fit(df_train)
        df_result = df_test.copy()
        df_result["prediction_perc"] = self.predict(df_test)
        df_result["prediction_perc"] = df_result["prediction_perc"].clip(lower=0)
        return df_result


# ---------------------------------------------------------
# Base ML Class for the other two baselines
# ---------------------------------------------------------
class BaseXGBBaseline:
    """Parent class for ML-based baselines to ensure a consistent API."""
    def __init__(self, features, target_name="perc_affected_pop_grid_region", **kwargs):
        self.features = features
        self.target_name = target_name
        self.model = XGBRegressor(
            booster="gbtree", 
            n_estimators=100, 
            learning_rate=0.01,
            max_depth=4,
            random_state=42,
            verbosity=0,
            **kwargs
        )
        
    def fit(self, X, y=None):
        target = y if y is not None else X[self.target_name]
        self.model.fit(X[self.features], target)
        return self
        
    def predict(self, X):
        return self.model.predict(X[self.features])
        
    def train_and_predict(self, df_train, df_test):
        self.fit(df_train)
        df_result = df_test.copy()
        df_result["prediction_perc"] = self.predict(df_test)
        df_result["prediction_perc"] = df_result["prediction_perc"].clip(lower=0)
        return df_result


# ---------------------------------------------------------
# 2. Windspeed Exposed Model
# ---------------------------------------------------------
class WindspeedExposedModel(BaseXGBBaseline):
    """'windspeed-exposed' baseline: Relies purely on wind speed and population exposure."""
    def __init__(self, target_name="perc_affected_pop_grid_region"):
        super().__init__(features=["wind_speed", "population"], target_name=target_name)


# ---------------------------------------------------------
# 3. Windspeed Historical Model
# ---------------------------------------------------------
class WindspeedHistoricalModel(BaseXGBBaseline):
    """'windspeed-historical' baseline: Combines wind speed exposure and historical impacts."""
    def __init__(self, target_name="perc_affected_pop_grid_region"):
        super().__init__(features=["wind_speed", "population", "N_events_5_years"], target_name=target_name)