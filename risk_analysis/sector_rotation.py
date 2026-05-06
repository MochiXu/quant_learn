"""Sector rotation strategy on Chinese sector ETFs.

Mirrors the OpenBB SPDR XL-series example, swapped for 中证/上证一级行业 ETFs.

Strategy (monthly):
  1. At month end t, rank sectors by their mean monthly return over the past
     `lookback_period` months (inclusive of t).
  2. Hold an equal-weighted basket of the `top_n` winners for month t+1.
  3. Compare cumulative return vs an equal-weighted market benchmark of all
     sectors in the universe.

Data source: hfq (后复权) ETF close prices saved by `downloader.fund` to
`data/raw/akshare/fund/hfq/{symbol}.parquet`. Codes not yet downloaded are
skipped with a warning.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.constant import AdjustType, SECTOR_ETFS
from common.dataloader import load_sector_etf_prices
from common.draw import configure_chinese_font

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "risk_analysis"


@dataclass
class RotationResult:
    # 月度收益率序列
    monthly_strategy_return: pd.Series  # 轮动策略：每月持有 top_n 赢家的等权收益
    monthly_equal_weight_return: pd.Series  # 等权策略：每月持有 universe 全部 ETF 的等权收益
    # 累计净值曲线（起点 = 1.0）
    strategy_cumulative: pd.Series
    equal_weight_cumulative: pd.Series
    # 终值与年化夏普
    sharpe_ratio: float
    final_strategy_return: float
    final_equal_weight_return: float


# 获取自定义的轮动组合收益率
def sector_rotation_strategy(
    prices: pd.DataFrame,
    lookback_period: int = 3,
    top_n: int = 3,
) -> pd.Series:
    """Equal-weight monthly rotation: long top_n by mean monthly return.

    Returns a Series of monthly portfolio returns indexed at month end.
    Each value at month m is the realized return earned during month m by
    holding the basket selected at the end of month m-1.
    """
    if top_n > prices.shape[1]:
        raise ValueError(f"top_n={top_n} exceeds number of ETFs={prices.shape[1]}")
    
    # 获取每个月的收盘价
    monthly_close = prices.resample("ME").last()
    # 计算每个月的收益率
    monthly_returns = monthly_close.pct_change(fill_method=None)

    # 用于存储自定义的轮动组合收益率
    portfolio = pd.Series(
        index=monthly_returns.index, dtype=float, name="portfolio_return"
    )

    # Need at least `lookback_period` past returns to rank, then we pick winners
    # at month i and earn the return realized in month i+1.
    for i in range(lookback_period, len(monthly_returns) - 1):
        window = monthly_returns.iloc[i - lookback_period + 1 : i + 1]
        # Drop ETFs that are not yet listed across the entire window.
        ranking = window.dropna(axis=1).mean()
        if ranking.empty:
            continue
        # 获取排名前 top_n 的 ETF 代码
        winners = ranking.nlargest(min(top_n, len(ranking))).index
        next_date = monthly_returns.index[i + 1]
        # 计算排名前 top_n 的 ETF 的平均收益率
        portfolio.loc[next_date] = monthly_returns.loc[next_date, winners].mean()

    # 返回自定义的轮动组合收益率，并删除缺失值
    return portfolio.dropna()


# 等权策略（基准）：每月等权重持有 universe 内全部 ETF
# 跟 sector_rotation_strategy 配对，作为「不主动做轮动」的对照组
def equal_weight_strategy(prices: pd.DataFrame) -> pd.Series:
    """等权策略的月收益序列。

    步骤：
      1. resample("ME") 取每月最后一个交易日的收盘价；
      2. pct_change 算出每月收益率（每只 ETF 一列）；
      3. mean(axis=1) 横向求均值 = 等权持仓的当月组合收益。

    某月有 ETF 还没上市时，那一列在该行是 NaN，mean(axis=1) 会自动跳过它，
    所以早期样本（只有 3-4 只 ETF 在交易时）也能正常算等权均值。
    """
    # 每月最后一个交易日的收盘价
    monthly_close = prices.resample("ME").last()
    # 每只 ETF 的月度收益率
    monthly_returns = monthly_close.pct_change(fill_method=None)
    # 横向取均值 = 等权持仓的当月组合收益
    return monthly_returns.mean(axis=1).dropna().rename("equal_weight_return")


def evaluate(
    strategy_returns: pd.Series,
    equal_weight_returns: pd.Series,
) -> RotationResult:
    """把月收益率序列汇总成 RotationResult：累计净值曲线 + 年化夏普。

    - 累计净值：把每月的「乘数」(1 + r) 逐月累乘，得到从初始 1.0 出发的净值曲线。
      例：[+5%, -3%, +2%] → 1.0 × 1.05 × 0.97 × 1.02 ≈ 1.039 → 这段时间总共涨 3.9%。
    - 年化夏普：(月均收益 / 月标准差) × √12。把月频指标年化用 √12，参考 risk_return.md。
      这里没扣无风险利率，是「裸夏普」；要严谨可以再减 r_f / 12。
    """
    # 累计净值 = (1 + 每月收益率) 的逐月累乘；起点为 1.0
    strategy_cum = (1 + strategy_returns).cumprod()
    equal_weight_cum = (1 + equal_weight_returns).cumprod()

    # 夏普率：方差太小（接近 0）时回退到 NaN，避免除 0
    if strategy_returns.std() > 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(12)
    else:
        sharpe = float("nan")

    return RotationResult(
        monthly_strategy_return=strategy_returns,
        monthly_equal_weight_return=equal_weight_returns,
        strategy_cumulative=strategy_cum,
        equal_weight_cumulative=equal_weight_cum,
        sharpe_ratio=float(sharpe),
        final_strategy_return=float(strategy_cum.iloc[-1]),
        final_equal_weight_return=float(equal_weight_cum.iloc[-1]),
    )


def plot_strategy_vs_benchmark(result: RotationResult, output_path: Path) -> None:
    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(
        result.strategy_cumulative.index,
        result.strategy_cumulative.values,
        label=f"行业轮动策略（终值 {result.final_strategy_return:.2f}，夏普 {result.sharpe_ratio:.2f}）",
        color="green",
        linewidth=1.8,
    )
    ax.plot(
        result.equal_weight_cumulative.index,
        result.equal_weight_cumulative.values,
        label=f"等权策略（终值 {result.final_equal_weight_return:.2f}）",
        color="steelblue",
        linewidth=1.8,
    )
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("行业轮动策略 vs 等权策略", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计净值（起始 = 1.0）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def run(
    lookback_period: int = 3,
    top_n: int = 3,
    adjust_type: AdjustType = "hfq",
    etfs: dict[str, dict[str, str]] | None = None,
    output_path: Path | None = None,
) -> RotationResult:
    """End-to-end: load ETFs → rotate → benchmark → evaluate → plot."""
    if etfs is None:
        etfs = SECTOR_ETFS
    prices = load_sector_etf_prices(etfs, adjust_type=adjust_type)
    strategy_ret = sector_rotation_strategy(
        prices, lookback_period=lookback_period, top_n=top_n
    )
    equal_ret = equal_weight_strategy(prices)

    # 对齐两条收益序列的起点：
    # - 等权策略从最早能算 pct_change 的那个月（约第 2 个月）就有值
    # - 轮动策略要先攒够 lookback_period 个月历史才能开始排名 → 起点更晚
    # 取「共同存在」的时间索引切片，保证两条累计净值都从同一个起点 1.0 出发，
    # 才能直接比较终值高低。否则一条 2014-01 起步、一条 2014-04 起步，没法公平对比。
    common = strategy_ret.index.intersection(equal_ret.index)
    result = evaluate(strategy_ret.loc[common], equal_ret.loc[common])

    if output_path is None:
        output_path = OUTPUT_DIR / f"sector_rotation_lb{lookback_period}_top{top_n}.png"
    plot_strategy_vs_benchmark(result, output_path)
    return result


if __name__ == "__main__":
    # res = run(lookback_period=3, top_n=3)
    # 通过网格敏感性, 可以算出来 12, 1 是最优的
    res = run(lookback_period=12, top_n=1)
    print(f"Final cumulative return — strategy:     {res.final_strategy_return:.2f}")
    print(f"Final cumulative return — equal weight: {res.final_equal_weight_return:.2f}")
    print(f"Strategy monthly Sharpe (annualized):   {res.sharpe_ratio:.2f}")
