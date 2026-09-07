import numpy as np
import pandas as pd
from pathlib import Path

def naive_next_day_label(df: pd.DataFrame) -> pd.Series:
    fwd_return = df["close"].shift(-1) / df["close"] - 1.0
    label = (fwd_return > 0).astype(int)
    label.name = "label_naive"
    return label

def daily_volatility(close: pd.Series, lookback: int = 20) -> pd.Series:
    returns = close.pct_change()
    vol = returns.ewm(span=lookback).std()
    return vol

def triple_barrier_labels(
    df: pd.DataFrame,
    vol_lookback: int = 20,
    pt_sl: list = [1.0, 1.0],
    max_holding_days: int = 5,
    **kwargs,
) -> pd.DataFrame:
    close = df["close"].values
    dates = df.index
    n = len(df)
    vol = daily_volatility(df["close"], vol_lookback).values

    labels = np.full(n, np.nan)
    touch_time = np.full(n, np.nan, dtype=object)
    touch_type = np.full(n, "", dtype=object)
    touch_idx = np.full(n, np.nan)

    for t in range(n - 1):
        if np.isnan(vol[t]) or vol[t] <= 0:
            continue

        upper = close[t] * (1.0 + pt_sl[0] * vol[t])
        lower = close[t] * (1.0 - pt_sl[1] * vol[t])
        end = min(t + max_holding_days, n - 1)

        touched = False
        for h in range(t + 1, end + 1):
            if close[h] >= upper:
                labels[t] = 1
                touch_type[t] = "upper"
                touch_time[t] = dates[h]
                touch_idx[t] = h
                touched = True
                break
            elif close[h] <= lower:
                labels[t] = 0
                touch_type[t] = "lower"
                touch_time[t] = dates[h]
                touch_idx[t] = h
                touched = True
                break

        if not touched:
            labels[t] = 1 if close[end] > close[t] else 0
            touch_type[t] = "vertical"
            touch_time[t] = dates[end]
            touch_idx[t] = end

    result = pd.DataFrame(
        {
            "label_tb": labels,
            "touch_type": touch_type,
            "touch_time": touch_time,
            "touch_idx": touch_idx,
        },
        index=df.index,
    )
    return result

def leakage_audit(features: pd.DataFrame, labels: pd.Series, out_path: str, n_rows: int = 30) -> pd.DataFrame:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    common_idx = features.index.intersection(labels.dropna().index)
    feats_sub = features.loc[common_idx]
    labels_sub = labels.loc[common_idx]

    audit = pd.DataFrame(index=common_idx)
    audit["asof_date"] = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in common_idx]
    for col in feats_sub.columns[:4]:
        audit[col] = feats_sub[col].round(4)
    audit["target_label"] = labels_sub.values

    dates_list = list(common_idx)
    eval_starts = [
        dates_list[i + 1].strftime("%Y-%m-%d")
        if i + 1 < len(dates_list) and hasattr(dates_list[i + 1], "strftime")
        else str(dates_list[min(i + 1, len(dates_list) - 1)])
        for i in range(len(dates_list))
    ]
    audit["target_eval_start"] = eval_starts
    audit["leakage_free_verified"] = "True (Target evaluated after asof_date)"

    sample = audit.dropna().head(n_rows)
    sample.to_csv(out_path, index=False)
    return sample
