import pandas as pd
import numpy as np

def persistence_baseline(df: pd.DataFrame) -> pd.Series:
    today_direction = (df["close"] > df["close"].shift(1)).astype(int)
    today_direction.name = "pred_persistence"
    return today_direction

def majority_class_baseline(labels: pd.Series, train_idx) -> pd.Series:
    if isinstance(train_idx, (range, slice)):
        train_labels = labels.iloc[train_idx]
    elif hasattr(train_idx, "dtype") and np.issubdtype(train_idx.dtype, np.integer):
        train_labels = labels.iloc[train_idx]
    elif isinstance(train_idx, (list, tuple)) and len(train_idx) > 0 and isinstance(train_idx[0], (int, np.integer)):
        train_labels = labels.iloc[train_idx]
    else:
        try:
            train_labels = labels.loc[train_idx]
        except (KeyError, TypeError):
            train_labels = labels.iloc[train_idx]
    majority_val = int(train_labels.mode()[0])
    pred = pd.Series(majority_val, index=labels.index, name="pred_majority")
    return pred
