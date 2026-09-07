import lightgbm as lgb
import numpy as np
import pandas as pd

def build_meta_dataset(oof_tcn_probs: np.ndarray, tabular_features: pd.DataFrame) -> pd.DataFrame:
    meta_X = tabular_features.copy()
    meta_X["tcn_prob"] = oof_tcn_probs
    return meta_X

def train_meta_model(meta_X_train, y_train, params: dict = None) -> lgb.LGBMClassifier:
    default_params = {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    if "verbose" not in default_params:
        default_params["verbose"] = -1

    model = lgb.LGBMClassifier(**default_params)
    model.fit(meta_X_train, y_train)
    return model
