"""双均线（dual moving average）动量策略回测。

参照 OpenBB 的 momentum_strategy 例子，对单只资产做：
  1. 计算 short_ma / long_ma 两条均线（简单移动平均，SMA）
  2. signal: 短均线在长均线之上 → +1（多头/动量向上）；之下 → -1（动量向下）
  3. position = signal.shift(1)，把信号往后挪一天，避免「今日收盘看到信号 → 今日收盘成交」的未来函数
  4. strategy_return = position × daily_return（持多头时跟着市场涨跌；持空头时反向）
  5. 累计净值 = (1 + return).cumprod()，跟买入持有的累计净值对比

接口风格与 cumulative_returns / sharpe_ratio 等模块对齐：
  - calculate_*  纯计算
  - draw_*       画图
  - calculate_and_draw_*  端到端、按 combined_prices 逐列跑、批量出图
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.constant import ADJUST_TYPE_MAP, POOL, AssetType, AdjustType
from common.draw import configure_chinese_font


# 计算双均线和仓位信号
def calculate_dual_ma_signal(
    prices: pd.Series,
    short_window: int,
    long_window: int,
    long_only: bool = True,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """对单只资产计算 short_ma / long_ma / signal / position。

    - 相对分歧 divergence = (short_ma - long_ma) / long_ma
      表示短均线相对于长均线偏离了多少（百分数）
    - signal 按 divergence 与 ±threshold 的关系给三态：
        divergence > +threshold       → +1（短均线明显在上方，多头）
        divergence < -threshold       → long_only=False → -1（做空）
                                        long_only=True  → 0（空仓）
        其余（|divergence| ≤ threshold） → 0（震荡区/盘整，空仓避开 whipsaw 假信号）
      threshold=0 退回标准双均线（除了 short_ma == long_ma 的零测度边界）
      threshold>0 + long_only=False → 真正的 {-1, 0, +1} 三态信号
      前 short_window 天数据不足，信号置 0
    - position: signal.shift(1) 避免未来函数（今日收盘看到信号 → 明日交易）
    """
    df = pd.DataFrame(index=prices.index)
    df["close"] = prices
    # 简单移动平均（SMA），min_periods=1 让前期窗口不足时也有值（用现有可见的最长窗口）
    df["short_ma"] = prices.rolling(window=short_window, min_periods=1).mean()
    df["long_ma"] = prices.rolling(window=long_window, min_periods=1).mean()

    # 相对分歧：threshold 也是相对值（如 0.02 = 短均线偏离长均线超过 2%）
    divergence = (df["short_ma"] - df["long_ma"]) / df["long_ma"]
    df["divergence"] = divergence

    # 信号：默认 0（暖机期不交易）；short_window 之后按 ±threshold 给三态
    df["signal"] = 0
    if len(df) > short_window:
        bull = divergence.iloc[short_window:] > threshold
        bear = divergence.iloc[short_window:] < -threshold
        # long_only 模式下空头位置改成 0，避开 A 股不能裸卖空的现实约束
        bear_value = 0 if long_only else -1
        signal_values = np.where(bull, 1, np.where(bear, bear_value, 0))
        df.iloc[short_window:, df.columns.get_loc("signal")] = signal_values

    # 仓位 = 信号往后挪一天，避免未来函数
    df["position"] = df["signal"].shift(1)
    return df


@dataclass
class BacktestResult:
    # 完整中间过程：close / short_ma / long_ma / signal / position
    #              / daily_return / strategy_return
    #              / cumulative_market_return / cumulative_strategy_return
    #              / portfolio_value
    data: pd.DataFrame
    # 终值（小数形式：0.50 = +50%；-0.20 = -20%）
    final_market_return: float
    final_strategy_return: float
    # 资金曲线终值（initial_capital × cumulative_strategy_return[-1]）
    final_portfolio_value: float


# 回测核心：对单只资产产出策略 vs 买入持有的对比序列与终值
def backtest_dual_ma(
    prices: pd.Series,
    short_window: int = 20,
    long_window: int = 50,
    initial_capital: float = 10000.0,
    long_only: bool = True,
    threshold: float = 0.0,
) -> BacktestResult:
    """双均线动量策略的单标的回测。"""
    df = calculate_dual_ma_signal(
        prices, short_window, long_window,
        long_only=long_only, threshold=threshold,
    )

    # 当日涨跌幅；首日无前一日 → NaN
    df["daily_return"] = df["close"].pct_change()

    # 策略当日收益 = 仓位 × 当日涨跌幅
    # 仓位为 NaN 或 0 时（暖机期）策略收益为 0/NaN，相当于空仓
    df["strategy_return"] = df["position"] * df["daily_return"]

    # 累计净值：起点 1.0，逐日累乘 (1 + 当日收益)
    # NaN 先填 0（首日 / 暖机期不动）保证净值从 1.0 平稳起步
    df["cumulative_market_return"] = (1 + df["daily_return"].fillna(0)).cumprod()
    df["cumulative_strategy_return"] = (1 + df["strategy_return"].fillna(0)).cumprod()

    # 资金曲线 = 初始资金 × 策略累计净值
    df["portfolio_value"] = initial_capital * df["cumulative_strategy_return"]

    return BacktestResult(
        data=df,
        final_market_return=float(df["cumulative_market_return"].iloc[-1] - 1.0),
        final_strategy_return=float(df["cumulative_strategy_return"].iloc[-1] - 1.0),
        final_portfolio_value=float(df["portfolio_value"].iloc[-1]),
    )


def draw_backtest(
    result: BacktestResult,
    symbol: str,
    name: str,
    asset_type: AssetType,
    adjust_type: AdjustType,
    short_window: int,
    long_window: int,
    output_path: Path,
    long_only: bool = True,
    threshold: float = 0.0,
) -> None:
    configure_chinese_font()
    df = result.data

    mode_label = "仅做多" if long_only else "多空双向"
    thr_label = f"，阈值 {threshold * 100:.1f}%" if threshold > 0 else ""
    strategy_label = f"双均线策略 [{mode_label}{thr_label}]"

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(
        df.index,
        df["cumulative_market_return"],
        label=(
            f"买入持有（终值 {df['cumulative_market_return'].iloc[-1]:.2f}，"
            f"累计 {result.final_market_return * 100:+.1f}%）"
        ),
        color="steelblue",
        linewidth=1.6,
    )
    ax.plot(
        df.index,
        df["cumulative_strategy_return"],
        label=(
            f"{strategy_label}（终值 {df['cumulative_strategy_return'].iloc[-1]:.2f}，"
            f"累计 {result.final_strategy_return * 100:+.1f}%）"
        ),
        color="green",
        linewidth=1.6,
    )
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(
        f"{symbol} {name} — 双均线 ({short_window}/{long_window}) {mode_label}{thr_label} 回测 "
        f"[{asset_type} {ADJUST_TYPE_MAP[adjust_type]}]",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("日期")
    ax.set_ylabel("累计净值（起始 = 1.0）")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# 端到端：对 combined_prices 每一列（每只资产）跑回测并出图
def calculate_and_draw_dual_ma_backtest(
    combined_prices: pd.DataFrame,
    asset_type: AssetType,
    adjust_type: AdjustType,
    output_dir: Path,
    short_window: int = 20,
    long_window: int = 50,
    initial_capital: float = 10000.0,
    long_only: bool = True,
    threshold: float = 0.0,
) -> dict[str, BacktestResult]:
    """对 combined_prices 每一列跑双均线回测，逐个保存图像。

    combined_prices 的列名是中文名（dataloader 设置的），用 POOL 反查 symbol
    给输出文件命名。

    long_only=True（默认）：短均线在长均线下方时空仓，符合 A 股个股不能裸卖空的现实。
    long_only=False：短均线在下方时反向做空（OpenBB 原版），便于学术对照。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 中文名 → symbol 的反查表（dataloader 的 close_series.name = meta["name"]）
    name_to_symbol = {meta["name"]: code for code, meta in POOL[asset_type].items()}

    # 文件名后缀区分模式 + 阈值，避免不同配置互相覆盖
    mode_suffix = "longonly" if long_only else "longshort"
    thr_suffix = f"_thr{int(threshold * 100)}pct" if threshold > 0 else ""

    results: dict[str, BacktestResult] = {}
    for name in combined_prices.columns:
        prices = combined_prices[name].dropna()
        # 数据少于长窗 + 2 行没法回测（至少要算出一个 long_ma + 一个 pct_change）
        if len(prices) < long_window + 2:
            print(f"[skip] {name}: too few rows ({len(prices)}) for long_window={long_window}")
            continue

        symbol = name_to_symbol.get(name, name)
        result = backtest_dual_ma(
            prices,
            short_window=short_window,
            long_window=long_window,
            initial_capital=initial_capital,
            long_only=long_only,
            threshold=threshold,
        )

        output_path = (
            output_dir
            / f"{asset_type}_{symbol}_{name}_{adjust_type}_ma{short_window}_{long_window}_{mode_suffix}{thr_suffix}.png"
        )
        draw_backtest(
            result,
            symbol=symbol,
            name=name,
            asset_type=asset_type,
            adjust_type=adjust_type,
            short_window=short_window,
            long_window=long_window,
            output_path=output_path,
            long_only=long_only,
            threshold=threshold,
        )

        results[symbol] = result
        print(
            f"  {symbol} {name}: "
            f"market {result.final_market_return * 100:+.2f}% / "
            f"strategy {result.final_strategy_return * 100:+.2f}% / "
            f"portfolio {result.final_portfolio_value:,.0f}"
        )

    return results
