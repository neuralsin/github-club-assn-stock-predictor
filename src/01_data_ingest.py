import os
from pathlib import Path
import pandas as pd
import yfinance as yf

def fetch_ohlcv(ticker: str, start: str, end: str, cache_dir: str = "data/raw") -> pd.DataFrame:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    clean_ticker = ticker.replace(".", "_").replace("^", "")
    cache_path = Path(cache_dir) / f"{clean_ticker}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if not df.empty:
            return df

    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="first")]
    df.to_parquet(cache_path)
    return df

if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    data = fetch_ohlcv(cfg["ticker"], cfg["start_date"], cfg["end_date"])
    print(data.shape, data.index.min(), data.index.max())
