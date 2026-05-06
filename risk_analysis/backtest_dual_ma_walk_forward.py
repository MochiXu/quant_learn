"""双均线 walk-forward 回测：用历史数据选参 → 在未来数据上验证。

为什么需要 walk-forward？
  backtest_dual_ma_sweep 用全样本数据扫描参数挑最优，本质是「拿答案对答案」——
  你已经看到了未来的全部走势，所以能挑到最优参数。这种 in-sample 业绩高估很严重，
  实战里你不可能提前知道哪对参数会最好。

walk-forward 的诚实做法：
  1. 用最早 train_years 年数据扫描参数 → 选出最优 (M, N)
  2. 在接下来 test_years 年用这对参数跑回测，拿到 OOS（out-of-sample）收益
  3. 滚动前进 test_years，下一段重新选参 → 再做一次 OOS 评估
  4. 拼接所有 OOS 段的策略收益 → 这才是「真实可执行的策略」业绩曲线

只要每段 train 都不偷看未来，最终 OOS 曲线就反映了你「真的能拿到的收益」。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.constant import ADJUST_TYPE_MAP, POOL, AssetType, AdjustType
from common.draw import configure_chinese_font
from risk_analysis.backtest_dual_ma import backtest_dual_ma
from risk_analysis.backtest_dual_ma_sweep import (
    find_best_window_pair,
    sweep_dual_ma_windows,
)


@dataclass
class WalkForwardResult:
    # 每段训练-测试的元数据 DataFrame
    # 列：train_start/end, test_start/end, best_short, best_long,
    #     in_sample_return, oos_strategy_return, oos_market_return
    segments: pd.DataFrame
    # OOS 拼接的累计净值曲线（起点 = 1.0）
    oos_strategy_cumulative: pd.Series
    oos_market_cumulative: pd.Series
    # OOS 拼接的日收益序列（用于进一步计算夏普等指标）
    oos_strategy_returns: pd.Series
    oos_market_returns: pd.Series
    # 终值（小数：0.50 = +50%）
    final_oos_strategy_return: float
    final_oos_market_return: float


def walk_forward_dual_ma(
    prices: pd.Series,
    short_range: tuple[int, int] = (10, 50),
    long_range: tuple[int, int] = (100, 200),
    short_step: int = 5,
    long_step: int = 10,
    train_years: int = 4,
    test_years: int = 1,
    initial_capital: float = 10000.0,
    long_only: bool = True,
    threshold: float = 0.0,
) -> WalkForwardResult:
    """单只资产的 walk-forward 回测。

    每段 train 都重新扫参数，用选出的最优 (M, N) 在紧接着的 test 段跑 OOS。
    test 段的 SMA 需要前面的历史做暖机，所以用 train+test 一起算后再切片 test 部分。
    """
    prices = prices.sort_index().dropna()

    segments_meta: list[dict] = []
    oos_strategy_returns: list[pd.Series] = []
    oos_market_returns: list[pd.Series] = []

    start = prices.index[0]
    end = prices.index[-1]

    cur_train_start = start
    while True:
        train_end = cur_train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        # 训练段（不包含 train_end 当天）
        train_mask = (prices.index >= cur_train_start) & (prices.index < train_end)
        train_slice = prices[train_mask]
        # train + test 联合段：跑回测时，SMA 在 train 段已经稳定，避免 test 起点 SMA 全是 NaN
        full_mask = (prices.index >= cur_train_start) & (prices.index < test_end)
        full_slice = prices[full_mask]

        n_train = len(train_slice)
        n_test = len(full_slice) - n_train

        # 训练数据要够支持最长的 long_window + 30 天 buffer，测试段至少 1 天
        if n_train < long_range[1] + 30 or n_test < 1:
            break

        # 1. In-sample sweep on train slice
        try:
            grid = sweep_dual_ma_windows(
                train_slice,
                short_range=short_range,
                long_range=long_range,
                short_step=short_step,
                long_step=long_step,
                long_only=long_only,
                threshold=threshold,
            )
            best_short, best_long, is_ret = find_best_window_pair(grid)
        except ValueError:
            # train 段全 NaN 或没有合法参数，停止
            break

        # 2. OOS apply: 在 train+test 上跑回测，切出 test 部分
        result = backtest_dual_ma(
            full_slice,
            short_window=best_short,
            long_window=best_long,
            initial_capital=initial_capital,
            long_only=long_only,
            threshold=threshold,
        )
        test_part = result.data.iloc[n_train:]

        oos_strategy_returns.append(test_part["strategy_return"])
        oos_market_returns.append(test_part["daily_return"])

        # 段汇总（终值 - 1 = 累计收益小数形式）
        seg_strat_ret = (1 + test_part["strategy_return"].fillna(0)).prod() - 1
        seg_mkt_ret = (1 + test_part["daily_return"].fillna(0)).prod() - 1
        segments_meta.append({
            "train_start": cur_train_start.date(),
            "train_end": train_end.date(),
            "test_start": test_part.index[0].date() if len(test_part) else train_end.date(),
            "test_end": test_part.index[-1].date() if len(test_part) else train_end.date(),
            "best_short": best_short,
            "best_long": best_long,
            "in_sample_return": round(is_ret, 4),
            "oos_strategy_return": round(seg_strat_ret, 4),
            "oos_market_return": round(seg_mkt_ret, 4),
        })

        # 滚动前进 test_years，下一段重新选参
        cur_train_start = cur_train_start + pd.DateOffset(years=test_years)
        if cur_train_start + pd.DateOffset(years=train_years) > end:
            break

    if not oos_strategy_returns:
        raise ValueError("Not enough data for walk-forward")

    oos_strat_ret = pd.concat(oos_strategy_returns)
    oos_mkt_ret = pd.concat(oos_market_returns)

    # 累计净值：用 fillna(0) 处理段间衔接处的暖机 NaN
    oos_strat_cum = (1 + oos_strat_ret.fillna(0)).cumprod()
    oos_mkt_cum = (1 + oos_mkt_ret.fillna(0)).cumprod()

    return WalkForwardResult(
        segments=pd.DataFrame(segments_meta),
        oos_strategy_cumulative=oos_strat_cum,
        oos_market_cumulative=oos_mkt_cum,
        oos_strategy_returns=oos_strat_ret,
        oos_market_returns=oos_mkt_ret,
        final_oos_strategy_return=float(oos_strat_cum.iloc[-1] - 1),
        final_oos_market_return=float(oos_mkt_cum.iloc[-1] - 1),
    )


def draw_walk_forward(
    result: WalkForwardResult,
    symbol: str,
    name: str,
    asset_type: AssetType,
    adjust_type: AdjustType,
    long_only: bool,
    output_path: Path,
    threshold: float = 0.0,
) -> None:
    """两图叠加：上图 OOS 累计净值对比；下图每段最优 (M, N) 看参数稳定性。"""
    configure_chinese_font()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # === 上图：OOS 累计净值 ===
    ax1.plot(
        result.oos_market_cumulative.index,
        result.oos_market_cumulative.values,
        label=(
            f"买入持有（终值 {result.oos_market_cumulative.iloc[-1]:.2f}，"
            f"{result.final_oos_market_return * 100:+.1f}%）"
        ),
        color="steelblue",
        linewidth=1.6,
    )
    ax1.plot(
        result.oos_strategy_cumulative.index,
        result.oos_strategy_cumulative.values,
        label=(
            f"OOS 策略（终值 {result.oos_strategy_cumulative.iloc[-1]:.2f}，"
            f"{result.final_oos_strategy_return * 100:+.1f}%）"
        ),
        color="green",
        linewidth=1.6,
    )

    # 用竖线标注每段 test 的起点，看参数切换的影响
    for _, row in result.segments.iterrows():
        ax1.axvline(
            pd.Timestamp(row["test_start"]),
            color="gray", linewidth=0.5, linestyle=":", alpha=0.5,
        )

    ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    mode_label = "仅做多" if long_only else "多空双向"
    thr_label = f"，阈值 {threshold * 100:.1f}%" if threshold > 0 else ""
    ax1.set_title(
        f"{symbol} {name} — Walk-Forward 双均线 [{mode_label}{thr_label}] "
        f"[{asset_type} {ADJUST_TYPE_MAP[adjust_type]}]",
        fontsize=13, fontweight="bold",
    )
    ax1.set_ylabel("OOS 累计净值（起始 = 1.0）")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # === 下图：每段最优 (M, N)，看参数随时间是否稳定 ===
    if not result.segments.empty:
        seg_dates = pd.to_datetime(result.segments["test_start"])
        ax2.plot(
            seg_dates, result.segments["best_short"],
            "o-", color="orange", label="best short_window (M)", markersize=6,
        )
        ax2.plot(
            seg_dates, result.segments["best_long"],
            "s-", color="purple", label="best long_window (N)", markersize=6,
        )
        ax2.set_xlabel("日期（test 段起点）")
        ax2.set_ylabel("最优均线长度（天）")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# 多变体合并图：把同一只资产的不同 (long_only, threshold) walk-forward 结果叠加显示
def draw_walk_forward_combined(
    variant_results: list[tuple[bool, float, WalkForwardResult]],
    full_prices: pd.Series,
    symbol: str,
    name: str,
    asset_type: AssetType,
    adjust_type: AdjustType,
    output_path: Path,
) -> None:
    """同一只资产、多个策略变体叠加在一张图里。

    画法：
      - 黑色粗线：买入持有，覆盖完整时间线（data start → end），起点 = 1.0
      - 彩色细线：每个变体的 OOS 策略累计净值，起点对齐到买入持有在 OOS 起点的水平
        （这样所有曲线在 OOS 起点处相交，之后的相对涨跌可以直接对比）
      - 红色竖线：OOS 起点（第一个 test 段开始），标注日期
    """
    if not variant_results:
        raise ValueError("variant_results is empty")

    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(13, 7))

    # 1. 买入持有覆盖完整时间线，起点 = 1.0
    full_market_cum = full_prices / full_prices.iloc[0]
    final_market_full = float(full_market_cum.iloc[-1] - 1.0)
    ax.plot(
        full_market_cum.index,
        full_market_cum.values,
        label=f"买入持有 base ({final_market_full * 100:+.1f}%)",
        color="black",
        linewidth=2.0,
        zorder=2,
    )

    # 2. OOS 起点：所有 variant 共享，取第一个的第一个 test 段起点
    oos_start = variant_results[0][2].oos_strategy_cumulative.index[0]
    # 策略起点需要锚定到买入持有在 OOS 起点的累计值，否则视觉上不接续
    market_at_oos = float(full_market_cum.loc[oos_start])

    # 3. 每个 variant 一条线
    palette = ["green", "steelblue", "darkorange", "purple", "brown", "magenta"]
    for i, (long_only, threshold, res) in enumerate(variant_results):
        mode_label = "仅做多" if long_only else "多空双向"
        thr_label = f" + 阈值 {threshold * 100:.1f}%" if threshold > 0 else ""
        legend = (
            f"{mode_label}{thr_label} "
            f"({res.final_oos_strategy_return * 100:+.1f}%)"
        )
        # 把 OOS 累计净值（起点 1.0）按买入持有在 OOS 起点的水平缩放，曲线视觉上接续
        strategy_aligned = res.oos_strategy_cumulative * market_at_oos
        ax.plot(
            strategy_aligned.index,
            strategy_aligned.values,
            label=legend,
            color=palette[i % len(palette)],
            linewidth=1.5,
            zorder=3,
        )

    # 4. OOS 起点竖线 + 文字标注
    ax.axvline(
        pd.Timestamp(oos_start),
        color="red",
        linewidth=1.3,
        linestyle="--",
        alpha=0.8,
        zorder=1,
    )
    ymin, ymax = ax.get_ylim()
    ax.text(
        pd.Timestamp(oos_start),
        ymin + (ymax - ymin) * 0.02,
        f"  OOS 起点 {pd.Timestamp(oos_start).strftime('%Y-%m-%d')}",
        fontsize=10, color="red", verticalalignment="bottom",
    )

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title(
        f"{symbol} {name} — Walk-Forward 双均线策略对照 "
        f"[{asset_type} {ADJUST_TYPE_MAP[adjust_type]}]",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("日期")
    ax.set_ylabel("累计净值（数据起点 = 1.0）")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def calculate_and_draw_dual_ma_walk_forward_combined(
    combined_prices: pd.DataFrame,
    asset_type: AssetType,
    adjust_type: AdjustType,
    output_dir: Path,
    variants: list[tuple[bool, float]] | None = None,
    short_range: tuple[int, int] = (10, 50),
    long_range: tuple[int, int] = (100, 200),
    short_step: int = 5,
    long_step: int = 10,
    train_years: int = 4,
    test_years: int = 1,
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    """对每只资产跑多个 (long_only, threshold) 变体，叠加到同一张 walk-forward 图里。

    variants 默认 4 个组合：
      [(True, 0.0), (False, 0.0), (True, 0.02), (False, 0.02)]
    分别对应 仅做多 / 多空双向 / 仅做多+2%阈值 / 多空双向+2%阈值。
    """
    if variants is None:
        variants = [(True, 0.0), (False, 0.0), (True, 0.02), (False, 0.02)]

    output_dir.mkdir(parents=True, exist_ok=True)
    name_to_symbol = {meta["name"]: code for code, meta in POOL[asset_type].items()}

    summary_rows: list[dict] = []
    for name in combined_prices.columns:
        prices = combined_prices[name].dropna()
        min_required = (train_years + test_years) * 240
        if len(prices) < min_required:
            print(f"[skip] {name}: too few rows ({len(prices)} < {min_required})")
            continue

        symbol = name_to_symbol.get(name, name)
        print(f"[{asset_type} {symbol} {name}] walk-forward combined ({len(variants)} variants)...")

        # 对每个 variant 跑 walk-forward
        variant_results: list[tuple[bool, float, WalkForwardResult]] = []
        for long_only, threshold in variants:
            try:
                res = walk_forward_dual_ma(
                    prices,
                    short_range=short_range,
                    long_range=long_range,
                    short_step=short_step,
                    long_step=long_step,
                    train_years=train_years,
                    test_years=test_years,
                    initial_capital=initial_capital,
                    long_only=long_only,
                    threshold=threshold,
                )
                variant_results.append((long_only, threshold, res))
                tag = f"{'longonly' if long_only else 'longshort'}, thr={threshold:.0%}"
                print(
                    f"  [{tag}] OOS strategy {res.final_oos_strategy_return * 100:+.2f}% / "
                    f"market {res.final_oos_market_return * 100:+.2f}%"
                )
            except ValueError as e:
                print(f"  skip ({long_only}, {threshold}): {e}")

        if not variant_results:
            continue

        # 画一张合并图
        out_path = (
            output_dir
            / f"{asset_type}_{symbol}_{name}_{adjust_type}_walkforward_combined.png"
        )
        draw_walk_forward_combined(
            variant_results,
            full_prices=prices,
            symbol=symbol,
            name=name,
            asset_type=asset_type,
            adjust_type=adjust_type,
            output_path=out_path,
        )

        # 收集 summary
        for long_only, threshold, res in variant_results:
            edge = res.final_oos_strategy_return - res.final_oos_market_return
            summary_rows.append({
                "symbol": symbol,
                "name": name,
                "long_only": long_only,
                "threshold": threshold,
                "n_segments": len(res.segments),
                "oos_strategy_return": round(res.final_oos_strategy_return, 4),
                "oos_market_return": round(res.final_oos_market_return, 4),
                "edge": round(edge, 4),
            })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        csv_path = (
            output_dir
            / f"summary_walkforward_combined_{asset_type}_{adjust_type}.csv"
        )
        summary.to_csv(csv_path, index=False)
        print(f"\nSaved summary: {csv_path}")

    return summary


def calculate_and_draw_dual_ma_walk_forward(
    combined_prices: pd.DataFrame,
    asset_type: AssetType,
    adjust_type: AdjustType,
    output_dir: Path,
    short_range: tuple[int, int] = (10, 50),
    long_range: tuple[int, int] = (100, 200),
    short_step: int = 5,
    long_step: int = 10,
    train_years: int = 4,
    test_years: int = 1,
    initial_capital: float = 10000.0,
    long_only: bool = True,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """对每只资产跑 walk-forward 并出图、保存段明细 CSV、汇总 CSV。

    Returns
    -------
    pd.DataFrame: 跨资产汇总（每行一只），列包括 OOS 策略/市场终值与 edge。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    name_to_symbol = {meta["name"]: code for code, meta in POOL[asset_type].items()}
    mode_suffix = "longonly" if long_only else "longshort"
    thr_suffix = f"_thr{int(threshold * 100)}pct" if threshold > 0 else ""

    summary_rows: list[dict] = []
    for name in combined_prices.columns:
        prices = combined_prices[name].dropna()
        # 数据要够 train + test（按交易日 ~252/年估算）
        min_required = (train_years + test_years) * 240
        if len(prices) < min_required:
            print(f"[skip] {name}: too few rows ({len(prices)} < {min_required})")
            continue

        symbol = name_to_symbol.get(name, name)
        print(f"[{asset_type} {symbol} {name}] walk-forward [{mode_suffix}{thr_suffix}]...")
        try:
            result = walk_forward_dual_ma(
                prices,
                short_range=short_range,
                long_range=long_range,
                short_step=short_step,
                long_step=long_step,
                train_years=train_years,
                test_years=test_years,
                initial_capital=initial_capital,
                long_only=long_only,
                threshold=threshold,
            )
        except ValueError as e:
            print(f"  skip {symbol}: {e}")
            continue

        # 累计净值 + 参数稳定性 双图
        out_path = (
            output_dir
            / f"{asset_type}_{symbol}_{name}_{adjust_type}_walkforward_{mode_suffix}{thr_suffix}.png"
        )
        draw_walk_forward(
            result,
            symbol=symbol,
            name=name,
            asset_type=asset_type,
            adjust_type=adjust_type,
            long_only=long_only,
            output_path=out_path,
            threshold=threshold,
        )

        # 段明细 CSV（看每段选了什么参数、in-sample vs OOS 收益差多少）
        seg_csv = (
            output_dir
            / f"{asset_type}_{symbol}_{name}_{adjust_type}_segments_{mode_suffix}{thr_suffix}.csv"
        )
        result.segments.to_csv(seg_csv, index=False)

        edge = result.final_oos_strategy_return - result.final_oos_market_return
        summary_rows.append({
            "symbol": symbol,
            "name": name,
            "n_segments": len(result.segments),
            "oos_strategy_return": round(result.final_oos_strategy_return, 4),
            "oos_market_return": round(result.final_oos_market_return, 4),
            "edge": round(edge, 4),
        })
        print(
            f"  OOS strategy {result.final_oos_strategy_return * 100:+.2f}% / "
            f"market {result.final_oos_market_return * 100:+.2f}% / "
            f"edge {edge * 100:+.2f}pp "
            f"({len(result.segments)} segments)"
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        csv_path = (
            output_dir
            / f"summary_walkforward_{asset_type}_{adjust_type}_{mode_suffix}{thr_suffix}.csv"
        )
        summary.to_csv(csv_path, index=False)
        print(f"\nSaved summary: {csv_path}")

    return summary
