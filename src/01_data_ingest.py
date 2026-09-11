import os
from pathlib import Path
import pandas as pd
import yfinance as yf

def load_csv_folder(folder_path: str, cache_dir: str = "data/raw", ticker: str = "TATAELXSI") -> pd.DataFrame:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) / f"{ticker}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if not df.empty:
            return df

    csv_files = sorted(Path(folder_path).glob("*.csv"))
    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)

    date_col = next((c for c in combined.columns if c.lower() in ["date", "datetime", "timestamp"]), combined.columns[0])
    extracted_date = combined[date_col].astype(str).str.extract(r"^([A-Za-z]{3}\s+[A-Za-z]{3}\s+\d+\s+\d{4}\s+\d{2}:\d{2}:\d{2})")[0]
    parsed_dates = pd.to_datetime(extracted_date, format="%a %b %d %Y %H:%M:%S", errors="coerce")
    if parsed_dates.isna().all():
        parsed_dates = pd.to_datetime(combined[date_col], errors="coerce")

    combined["datetime"] = parsed_dates
    combined = combined.dropna(subset=["datetime"])
    combined = combined.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    combined = combined.set_index("datetime")

    col_map = {c: c.strip().lower() for c in combined.columns}
    combined = combined.rename(columns=col_map)
    req_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in combined.columns]
    df = combined[req_cols].astype(float)
    df.to_parquet(cache_path)
    return df

def fetch_ohlcv(
    ticker: str,
    start: str = None,
    end: str = None,
    cache_dir: str = "data/raw",
    data_source: str = "yfinance",
    folder_path: str = None,
) -> pd.DataFrame:
    if data_source == "csv_folder" or folder_path:
        path = folder_path or "tataelxsi"
        return load_csv_folder(path, cache_dir, ticker)

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
    data = fetch_ohlcv(
        ticker=cfg["ticker"],
        start=cfg.get("start_date"),
        end=cfg.get("end_date"),
        data_source=cfg.get("data_source", "yfinance"),
        folder_path=cfg.get("folder_path"),
    )
    print(data.shape, data.index.min(), data.index.max())

