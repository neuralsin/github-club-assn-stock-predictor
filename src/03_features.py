import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from ta.momentum import RSIIndicator, StochasticOscillator
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
    conv = np.convolve(values, weights, mode="valid")
    out = np.full(len(values), np.nan)
    out[width - 1 :] = conv
    return pd.Series(out, index=series.index)

def find_min_ffd(series: pd.Series, d_grid: list, adf_pvalue_threshold: float = 0.05) -> float:
    test_series = series if len(series) <= 3000 else series.iloc[-3000:]
    for d in d_grid:
        diffed = frac_diff_fixed(test_series, d).dropna()
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
    close = df["close"]

    feats["raw_close"] = close
    feats["raw_return_1d"] = close.pct_change()
    feats["ret_3"] = close.pct_change(3)
    feats["ret_6"] = close.pct_change(6)
    feats["ret_12"] = close.pct_change(12)

    has_volume = "volume" in df.columns and df["volume"].notna().sum() > 0 and (df["volume"] > 0).any()
    if has_volume:
        feats["raw_volume"] = df["volume"]
        vol_mean = df["volume"].rolling(20).mean()
        vol_std = df["volume"].rolling(20).std().replace(0, np.nan)
        feats["volume_zscore_20d"] = (df["volume"] - vol_mean) / vol_std
    else:
        feats["raw_volume"] = 1.0

    has_ohlc = all(col in df.columns for col in ["open", "high", "low", "close"])
    if has_ohlc:
        hl_range = (df["high"] - df["low"]).replace(0, 1e-5)
        feats["candle_range"] = hl_range / close
        feats["candle_body"] = (close - df["open"]) / close
        feats["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / close
        feats["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / close
        feats["parkinson_vol"] = np.sqrt((np.log(df["high"] / df["low"]) ** 2).rolling(12).mean() / (4 * np.log(2)))
        stoch = StochasticOscillator(df["high"], df["low"], close, window=14, smooth_window=3)
        feats["stoch_k"] = stoch.stoch()
        feats["stoch_d"] = stoch.stoch_signal()

    for span in [5, 15, 30, 60]:
        ema = close.ewm(span=span).mean()
        feats[f"ema_dist_{span}"] = (close - ema) / ema

    optimal_d = find_min_ffd(close, d_grid, adf_pvalue_threshold)
    feats["fracdiff_close"] = frac_diff_fixed(close, optimal_d)

    feats["rsi_14"] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    feats["macd_diff"] = macd.macd_diff()
    bb = BollingerBands(close, window=20)
    band_range = bb.bollinger_hband() - bb.bollinger_lband()
    feats["bb_pctb"] = (close - bb.bollinger_lband()) / band_range.replace(0, np.nan)
    feats["volatility_20d"] = close.pct_change().rolling(20).std()

    if any(hasattr(d, "hour") and d.hour != 0 for d in df.index[:10]):
        minute_of_day = df.index.hour * 60 + df.index.minute
        feats["time_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
        feats["time_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))

    clean_feats = feats.dropna()
    return clean_feats, optimal_d

