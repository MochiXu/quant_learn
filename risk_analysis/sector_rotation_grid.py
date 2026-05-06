"""Sector rotation 参数敏感性网格扫描。

复用 sector_rotation.py 里的策略原语：
  - sector_rotation_strategy：动量轮动
  - equal_weight_strategy：等权对照组
  - evaluate：把月收益序列汇总成累计净值 + 夏普

本文件只负责对 (lookback_period, top_n) 多组合扫描，回答「默认 (3, 3) 是不是
最优？信号在哪个回看窗最稳定？」。结果可作为后续 walk-forward / heatmap 的输入。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.constant import AdjustType, SECTOR_ETFS
from common.dataloader import load_sector_etf_prices
from risk_analysis.sector_rotation import (
    OUTPUT_DIR,
    equal_weight_strategy,
    evaluate,
    sector_rotation_strategy,
)


def grid_sweep(
    prices: pd.DataFrame,
    lookback_periods: tuple[int, ...] = (1, 3, 6, 12),
    top_ns: tuple[int, ...] = (1, 2, 3, 5, 7),
) -> pd.DataFrame:
    """对 (lookback_period, top_n) 做网格扫描，每对参数算一遍策略 vs 等权对比。

    Returns
    -------
    pd.DataFrame，每行一个 (lookback, top_n) 组合，列包含：
      - strategy_final: 该组合下轮动策略的累计净值终值
      - equal_weight_final: 同期等权策略终值（对齐到策略起点，所以会随 lookback 不同而不同）
      - edge: strategy_final - equal_weight_final（正数为跑赢）
      - sharpe: 策略年化夏普
      - wins: edge > 0

    注意：equal_weight_final 不是一个常数。因为每个 lookback 下策略的起点不同，
    我们在 evaluate 里把等权序列也切到相同起点对齐，所以等权终值也跟着 lookback 变。
    这才是公平比较——「同样的时间段内，主动轮动是否比被动持有划算？」
    """
    rows: list[dict] = []
    # 等权策略只跟 prices 有关，跟 lookback/top_n 无关，提前算一次
    equal_ret_full = equal_weight_strategy(prices)

    n_etfs = prices.shape[1]
    for lb in lookback_periods:
        for tn in top_ns:
            if tn > n_etfs:
                # top_n 超过 universe 大小，跳过
                continue
            try:
                strat_ret = sector_rotation_strategy(
                    prices, lookback_period=lb, top_n=tn
                )
                # 把等权切到跟策略相同的时间窗口，保证公平对比
                common = strat_ret.index.intersection(equal_ret_full.index)
                result = evaluate(
                    strat_ret.loc[common], equal_ret_full.loc[common]
                )
                edge = result.final_strategy_return - result.final_equal_weight_return
                rows.append({
                    "lookback": lb,
                    "top_n": tn,
                    "strategy_final": round(result.final_strategy_return, 2),
                    "equal_weight_final": round(result.final_equal_weight_return, 2),
                    "edge": round(edge, 2),
                    "sharpe": round(result.sharpe_ratio, 2),
                    "wins": edge > 0,
                })
            except Exception as e:
                print(f"[grid] skip (lookback={lb}, top_n={tn}): {e}")

    return pd.DataFrame(rows)


def print_grid_sweep(grid: pd.DataFrame) -> None:
    """把 grid_sweep 的结果按表格形式打印，跑赢的组合用 ✓ 标记。"""
    header = (
        f"{'lookback':>9} {'top_n':>5} | "
        f"{'strat':>6} {'equal':>6} {'edge':>6} {'sharpe':>6} | wins?"
    )
    print(header)
    print("-" * len(header))
    for _, row in grid.iterrows():
        tag = "✓" if row["wins"] else " "
        print(
            f"{int(row['lookback']):>9} {int(row['top_n']):>5} | "
            f"{row['strategy_final']:>6.2f} {row['equal_weight_final']:>6.2f} "
            f"{row['edge']:>+6.2f} {row['sharpe']:>6.2f} |   {tag}"
        )


def run_grid_sweep(
    lookback_periods: tuple[int, ...] = (1, 3, 6, 12),
    top_ns: tuple[int, ...] = (1, 2, 3, 5, 7),
    adjust_type: AdjustType = "hfq",
    etfs: dict[str, dict[str, str]] | None = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """End-to-end 网格扫描：加载数据 → 跑所有 (lookback, top_n) → 打印 + 落盘。"""
    if etfs is None:
        etfs = SECTOR_ETFS
    prices = load_sector_etf_prices(etfs, adjust_type=adjust_type)
    grid = grid_sweep(prices, lookback_periods=lookback_periods, top_ns=top_ns)
    print_grid_sweep(grid)

    if save_csv:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = OUTPUT_DIR / "sector_rotation_grid_sweep.csv"
        grid.to_csv(csv_path, index=False)
        print(f"\nSaved grid to: {csv_path}")

    return grid


if __name__ == "__main__":
    run_grid_sweep()
