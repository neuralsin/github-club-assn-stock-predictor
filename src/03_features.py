import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

warnings.filterwarnings("ignore", category=FutureWarning)

def get_fracdiff_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])

def frac_diff_fixed(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    weights = get_fracdiff_weights(d, threshold)
    if len(weights) >= len(series):
        max_w = max(2, len(series) // 2)
        weights = weights[-max_w:]
    width = len(weights)
    values = series.values
    out = np.full(len(values), np.nan)
    for i in range(width - 1, len(values)):
        window = values[i - width + 1 : i + 1]
        out[i] = np.dot(weights, window)
    return pd.Series(out, index=series.index)

def find_min_ffd(series: pd.Series, d_grid: list, adf_pvalue_threshold: float = 0.05) -> float:
    for d in d_grid:
        diffed = frac_diff_fixed(series, d).dropna()
        if len(diffed) < 50:
            continue
        adf_res = adfuller(diffed, maxlag=1, regression="c", autolag=None)
        pvalue = float(adf_res[1])
        if pvalue < adf_pvalue_threshold:
            return float(d)
    return float(d_grid[-1])

def build_features(
    df: pd.DataFrame,
    d_grid: list = None,
    adf_pvalue_threshold: float = 0.05,
    threshold: float = 1e-4,
    **kwargs,
) -> tuple[pd.DataFrame, float]:
    if d_grid is None:
        d_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    feats = pd.DataFrame(index=df.index)

    feats["raw_close"] = df["close"]
    feats["raw_volume"] = df["volume"]
    feats["raw_return_1d"] = df["close"].pct_change()

    optimal_d = find_min_ffd(df["close"], d_grid, adf_pvalue_threshold)
    feats["fracdiff_close"] = frac_diff_fixed(df["close"], optimal_d)

    feats["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    feats["macd_diff"] = macd.macd_diff()
    bb = BollingerBands(df["close"], window=20)
    band_range = bb.bollinger_hband() - bb.bollinger_lband()
    feats["bb_pctb"] = (df["close"] - bb.bollinger_lband()) / band_range.replace(0, np.nan)
    feats["volatility_20d"] = df["close"].pct_change().rolling(20).std()
    
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std().replace(0, np.nan)
    feats["volume_zscore_20d"] = (df["volume"] - vol_mean) / vol_std

    clean_feats = feats.dropna()
    return clean_feats, optimal_d
