import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

class TwoStageXGBoost:
    """
    Custom 2-stage XGBoost model pipeline.
    Stage 1: Binary Classification (Impact > 0)
    Stage 2: Regression (Magnitude of impact, conditioned on predicted Impact > 0)
    """
    def __init__(self, clf_params=None, reg_params=None, features=None, target_name="perc_affected_pop_grid_region"):
        self.clf_params = clf_params or self._default_clf_params()
        self.reg_params = reg_params or self._default_reg_params()
        self.features = features
        self.target_name = target_name
        self.classifier = XGBClassifier(**self.clf_params)
        self.regressor = XGBRegressor(**self.reg_params)

    @staticmethod
    def _default_clf_params():
        return {
            "booster": "gbtree", "colsample_bytree": 0.8, "gamma": 0.5, 
            "learning_rate": 0.01, "max_depth": 4, "min_child_weight": 1, 
            "n_estimators": 100, "subsample": 0.8, "verbosity": 0, 
            "random_state": 0, "objective": "binary:logistic", "eval_metric": "logloss"
        }

    @staticmethod
    def _default_reg_params():
        return {
            "booster": "gbtree", "colsample_bytree": 0.8, "gamma": 0.5, 
            "learning_rate": 0.01, "max_depth": 4, "min_child_weight": 1, 
            "n_estimators": 100, "objective": "reg:squarederror", 
            "eval_metric": "rmse", "subsample": 0.8, "verbosity": 0, "random_state": 0
        }

    @staticmethod
    def oversample(df, target, u=1):
        """Oversample majority class using multiplier u."""
        minority_size = df[target].sum()
        majority_size = int(u * minority_size)
        
        df_balanced = (
            df.groupby(target, group_keys=False)
            .apply(lambda x: x.sample(n=min(minority_size if x.name == 1 else majority_size, len(x)), random_state=42))
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        return df_balanced

    def train_and_predict(self, df_train, df_test, u1=3, u2=3):
        """Executes the custom 2-stage training and prediction flow with oversampling."""
        # Stage 1: Classification
        df_train["reported_bin"] = (df_train[self.target_name] > 0).astype(int)
        df_test["reported_bin"] = (df_test[self.target_name] > 0).astype(int)
        
        df_train_balanced_1 = self.oversample(df_train, target="reported_bin", u=u1)
        self.classifier.fit(df_train_balanced_1[self.features], df_train_balanced_1["reported_bin"])
        
        y_proba = self.classifier.predict_proba(df_test[self.features])[:, 1]
        df_result = df_test.copy()
        df_result["predicted_proba"] = y_proba
        df_result["predicted_bin"] = (y_proba >= 0.5).astype(int)

        # Stage 2: Regression
        df_train_high = df_train[df_train[self.target_name] > 0].copy()
        df_train_high["impact_high"] = (df_train_high[self.target_name] >= 15).astype(int)
        
        df_train_balanced_2 = self.oversample(df_train_high, target="impact_high", u=u2)
        
        df_stage2_pos = df_result[df_result["predicted_bin"] == 1].copy()
        df_stage2_neg = df_result[df_result["predicted_bin"] == 0].copy()
        
        if not df_stage2_pos.empty:
            self.regressor.fit(df_train_balanced_2[self.features], df_train_balanced_2[self.target_name])
            df_stage2_pos["prediction_perc"] = self.regressor.predict(df_stage2_pos[self.features])
            
        df_stage2_neg["prediction_perc"] = 0
        
        return pd.concat([df_stage2_pos, df_stage2_neg], axis=0).sort_index()