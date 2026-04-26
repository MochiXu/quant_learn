"""
Quick smoke test: fetch Apple's recent stock data with OpenBB
and print basic NumPy stats.

Usage:
    conda activate myTools
    python scripts/hello_openbb.py
"""

from openbb import obb
import numpy as np

SYMBOL = "AAPL"

print(f"Fetching daily prices for {SYMBOL} ...")
result = obb.equity.price.historical(SYMBOL, provider="yfinance")
df = result.to_dataframe()

closes = df["close"].to_numpy()

print(f"\n--- {SYMBOL} Close Price Stats (last {len(closes)} trading days) ---")
print(f"  Mean:   {np.mean(closes):.2f}")
print(f"  Std:    {np.std(closes):.2f}")
print(f"  Min:    {np.min(closes):.2f}")
print(f"  Max:    {np.max(closes):.2f}")
print(f"  Latest: {closes[-1]:.2f}")

daily_returns = np.diff(closes) / closes[:-1]
print(f"\n--- Daily Returns ---")
print(f"  Mean:   {np.mean(daily_returns):.4%}")
print(f"  Std:    {np.std(daily_returns):.4%}")
print(f"  Sharpe (annualized, rf=0): {np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252):.2f}")
