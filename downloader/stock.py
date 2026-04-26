from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Literal

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TOKEN = os.environ["TUSHARE_TOKEN"]

AdjustType = Literal["raw", "qfq", "hfq"]

ADJUST_MAP: dict[AdjustType, str | None] = {
    "raw": None,
    "qfq": "qfq",
    "hfq": "hfq",
}


def to_ts_code(symbol: str) -> str:
    """Convert a symbol to tushare format.

    Pure numbers get .SH/.SZ suffix added.
    Already-suffixed codes like '600519.SH' are returned as-is.
    """
    if "." in symbol:
        return symbol.upper()
    if symbol.startswith("6") or symbol.startswith("5"):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def _normalize_tushare_df(
    df: pd.DataFrame,
    *,
    symbol: str,
    adjust_type: AdjustType,
) -> pd.DataFrame:
    rename_map = {
        "trade_date": "date",
        "pct_chg": "pct_change",
        "vol": "volume",
    }

    df = df.rename(columns=rename_map).copy()

    if "date" not in df.columns:
        raise ValueError(f"Missing date column for {symbol}. Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["symbol"] = symbol
    df["adjust"] = adjust_type
    df["source"] = "tushare"
    df["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    numeric_cols = [
        "open", "high", "low", "close", "pre_close",
        "change", "pct_change", "volume", "amount",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [
        "date", "symbol", "adjust",
        "open", "high", "low", "close", "pre_close",
        "change", "pct_change", "volume", "amount",
        "source", "updated_at",
    ]

    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols]
    df = df.sort_values("date")
    df = df.drop_duplicates(["date", "symbol", "adjust"], keep="last")
    df = df.reset_index(drop=True)

    return df


@dataclass
class StockDownloadParam:
    symbols: list[str] = field(default_factory=list)
    download_path: str = "data/raw/tushare/stock"
    store_csv: bool = True
    store_parquet: bool = True
    skip_exists: bool = True
    start_date: str = "20150101"
    end_date: str = "20260425"
    adjust_types: tuple[AdjustType, ...] = ("raw", "qfq", "hfq")
    max_retries: int = 3
    sleep_seconds: int = 8


class StockDownloader:
    def __init__(self, param: StockDownloadParam) -> None:
        self.param = param
        ts.set_token(TOKEN)
        self.pro = ts.pro_api(TOKEN)
        self.output_dir = Path(param.download_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_adjust_dir(self, adjust_type: AdjustType) -> Path:
        d = self.output_dir / adjust_type
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _fetch_raw(self, symbol: str, adjust_type: AdjustType) -> pd.DataFrame:
        ts_code = to_ts_code(symbol)
        return ts.pro_bar(
            ts_code=ts_code,
            api=self.pro,
            adj=ADJUST_MAP[adjust_type],
            asset="E",
            freq="D",
            start_date=self.param.start_date,
            end_date=self.param.end_date,
        )

    def _save(self, df: pd.DataFrame, symbol: str, adjust_type: AdjustType) -> None:
        d = self._get_adjust_dir(adjust_type)

        if self.param.store_parquet:
            df.to_parquet(d / f"{symbol}.parquet", index=False)

        if self.param.store_csv:
            df.to_csv(d / f"{symbol}.csv", index=False, encoding="utf-8-sig")

    def _should_skip(self, symbol: str, adjust_type: AdjustType) -> bool:
        if not self.param.skip_exists:
            return False
        d = self._get_adjust_dir(adjust_type)
        return (d / f"{symbol}.parquet").exists() or (d / f"{symbol}.csv").exists()

    def download_one(self, symbol: str, adjust_type: AdjustType) -> pd.DataFrame:
        last_error: Exception | None = None

        for attempt in range(1, self.param.max_retries + 1):
            try:
                print(f"[stock] Fetching {symbol} adjust={adjust_type} attempt={attempt}")

                raw_df = self._fetch_raw(symbol, adjust_type)

                if raw_df is None or raw_df.empty:
                    raise RuntimeError(f"No data returned for {symbol} adjust={adjust_type}")

                df = _normalize_tushare_df(raw_df, symbol=symbol, adjust_type=adjust_type)
                self._save(df, symbol, adjust_type)

                print(f"Saved {symbol} adjust={adjust_type}: {len(df)} rows")
                return df

            except Exception as e:
                last_error = e
                wait = self.param.sleep_seconds * attempt
                print(f"Failed {symbol} adjust={adjust_type} attempt={attempt}: {e}")
                print(f"Sleep {wait}s before retry...")
                sleep(wait)

        raise RuntimeError(
            f"Failed to fetch {symbol} adjust={adjust_type} "
            f"after {self.param.max_retries} retries"
        ) from last_error

    def download_all(self) -> dict[str, dict[str, pd.DataFrame]]:
        results: dict[str, dict[str, pd.DataFrame]] = {}

        for symbol in self.param.symbols:
            results[symbol] = {}
            for adjust_type in self.param.adjust_types:
                if self._should_skip(symbol, adjust_type):
                    print(f"Skip existing: {symbol} adjust={adjust_type}")
                    continue

                try:
                    df = self.download_one(symbol, adjust_type)
                    results[symbol][adjust_type] = df
                    sleep(self.param.sleep_seconds)
                except Exception as e:
                    sleep(self.param.sleep_seconds)
                    print(f"FAILED symbol={symbol} adjust={adjust_type}: {e}")

        return results
