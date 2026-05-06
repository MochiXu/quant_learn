"""双均线 (short_window, long_window) 二维参数扫描 + 最优参数回测。

对每只资产扫描 M × N 种 (short, long) 组合，找出策略累计收益最大的那一对，
然后用最优参数出两张图：
  1. 累计收益曲线（最优参数下的策略 vs 买入持有）—— 直接复用 draw_backtest
  2. 3D 曲面图：x=short_window, y=long_window, z=策略累计收益率（%）

接口风格与 backtest_dual_ma 对齐：calculate_* / draw_* / calculate_and_draw_*。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
import numpy as np
import pandas as pd

from common.constant import ADJUST_TYPE_MAP, POOL, AssetType, AdjustType
from common.draw import configure_chinese_font
from risk_analysis.backtest_dual_ma import BacktestResult, backtest_dual_ma, draw_backtest


# 对单只资产扫描 (short_window, long_window) 二维网格
def sweep_dual_ma_windows(
    prices: pd.Series,
    short_range: tuple[int, int] = (10, 50),
    long_range: tuple[int, int] = (100, 200),
    short_step: int = 5,
    long_step: int = 10,
    long_only: bool = True,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """对每个 (short, long) 组合跑一次回测，记录策略累计收益。

    Returns
    -------
    pd.DataFrame
        index = long_window 列表（升序）
        columns = short_window 列表（升序）
        values = 策略累计收益（小数：0.50 = +50%）
        非法组合（short >= long 或回测失败）→ NaN
    """
    short_windows = list(range(short_range[0], short_range[1] + 1, short_step))
    long_windows = list(range(long_range[0], long_range[1] + 1, long_step))

    grid = pd.DataFrame(
        index=long_windows,
        columns=short_windows,
        dtype=float,
    )

    for sw in short_windows:
        for lw in long_windows:
            # 短均线必须严格小于长均线，否则信号没有意义
            if sw >= lw:
                continue
            try:
                result = backtest_dual_ma(
                    prices,
                    short_window=sw,
                    long_window=lw,
                    long_only=long_only,
                    threshold=threshold,
                )
                grid.loc[lw, sw] = result.final_strategy_return
            except Exception as e:
                print(f"    skip (short={sw}, long={lw}): {e}")

    return grid


def find_best_window_pair(grid: pd.DataFrame) -> tuple[int, int, float]:
    """在网格里找策略累计收益最大的 (short_window, long_window, return)。"""
    # grid: index=long, columns=short → stack 后 idxmax 返回 (long, short)
    flat = grid.stack()
    if flat.empty or flat.isna().all():
        raise ValueError("Grid is empty or all-NaN")
    best_long, best_short = flat.idxmax()
    return int(best_short), int(best_long), float(flat.max())


# 3D 曲面图：M × N → 收益率
def draw_window_sweep_surface(
    grid: pd.DataFrame,
    symbol: str,
    name: str,
    asset_type: AssetType,
    adjust_type: AdjustType,
    long_only: bool,
    output_path: Path,
    best_short: int | None = None,
    best_long: int | None = None,
) -> None:
    """画 3D 曲面：x=short_window (M), y=long_window (N), z=策略累计收益率 (%)。

    最优点用红色五角星标出，方便一眼看到「峰值在哪里」。
    """
    configure_chinese_font()

    # X = short windows（列），Y = long windows（行）
    short_windows = grid.columns.values.astype(float)
    long_windows = grid.index.values.astype(float)
    X, Y = np.meshgrid(short_windows, long_windows)
    Z = grid.values.astype(float) * 100.0  # 转成百分数显示

    # NaN 用全局最小值填充，避免 plot_surface 报错或留洞
    Z_filled = np.where(np.isnan(Z), np.nanmin(Z), Z)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(projection="3d")

    surf = ax.plot_surface(
        X, Y, Z_filled,
        cmap="viridis",
        edgecolor="none",
        alpha=0.9,
        antialiased=True,
    )

    # 标出最优点
    if best_short is not None and best_long is not None:
        best_z = grid.loc[best_long, best_short] * 100.0
        ax.scatter(
            [best_short], [best_long], [best_z],
            color="red", marker="*", s=200, edgecolors="white", linewidths=1.5,
            label=f"最优 ({best_short}, {best_long}) → {best_z:.1f}%",
            zorder=10,
        )
        ax.legend(loc="upper left", fontsize=10)

    ax.set_xlabel("short_window (M)", fontsize=11, labelpad=8)
    ax.set_ylabel("long_window (N)", fontsize=11, labelpad=8)
    ax.set_zlabel("策略累计收益率 (%)", fontsize=11, labelpad=8)

    mode_label = "仅做多" if long_only else "多空双向"
    ax.set_title(
        f"{symbol} {name} — 双均线参数扫描 [{mode_label}] "
        f"[{asset_type} {ADJUST_TYPE_MAP[adjust_type]}]",
        fontsize=13, fontweight="bold", pad=14,
    )

    # 视角调一下，避免顶视看不出高低
    ax.view_init(elev=28, azim=-130)

    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=18, pad=0.08, label="收益率 (%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# 端到端：每只资产 → 扫网格 → 找最优 → 画两张图
def calculate_and_draw_dual_ma_sweep(
    combined_prices: pd.DataFrame,
    asset_type: AssetType,
    adjust_type: AdjustType,
    output_dir: Path,
    short_range: tuple[int, int] = (10, 50),
    long_range: tuple[int, int] = (100, 200),
    short_step: int = 5,
    long_step: int = 10,
    initial_capital: float = 10000.0,
    long_only: bool = False,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """对每只资产扫描参数网格，挑最优组合，每只资产输出两张图：
    一张是最优参数下的累计净值曲线，一张是 (M, N) → 收益率的 3D 曲面。

    Returns
    -------
    pd.DataFrame
        每行一只资产，列：symbol / name / best_short / best_long /
                         strategy_return / market_return / edge
        edge = strategy_return - market_return（正数 = 策略跑赢买入持有）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    name_to_symbol = {meta["name"]: code for code, meta in POOL[asset_type].items()}
    mode_suffix = "longonly" if long_only else "longshort"
    thr_suffix = f"_thr{int(threshold * 100)}pct" if threshold > 0 else ""

    summary_rows: list[dict] = []
    for name in combined_prices.columns:
        prices = combined_prices[name].dropna()
        # 数据要够算出最长的 long_window + 一个 pct_change
        if len(prices) < long_range[1] + 2:
            print(f"[skip] {name}: too few rows ({len(prices)}) for max_long={long_range[1]}")
            continue

        symbol = name_to_symbol.get(name, name)
        print(f"[{asset_type} {symbol} {name}] sweeping...")

        # 1. 扫描网格
        grid = sweep_dual_ma_windows(
            prices,
            short_range=short_range,
            long_range=long_range,
            short_step=short_step,
            long_step=long_step,
            long_only=long_only,
            threshold=threshold,
        )

        # 2. 挑最优
        try:
            best_short, best_long, best_strategy_ret = find_best_window_pair(grid)
        except ValueError as e:
            print(f"  skip {symbol}: {e}")
            continue

        # 3. 用最优参数完整跑一次回测，拿到 cumulative 序列用于画图
        best_result = backtest_dual_ma(
            prices,
            short_window=best_short,
            long_window=best_long,
            initial_capital=initial_capital,
            long_only=long_only,
            threshold=threshold,
        )

        # 4a. 累计收益图（复用 backtest_dual_ma 里的 draw_backtest）
        cum_path = (
            output_dir
            / f"{asset_type}_{symbol}_{name}_{adjust_type}_best_ma{best_short}_{best_long}_{mode_suffix}{thr_suffix}.png"
        )
        draw_backtest(
            best_result,
            symbol=symbol,
            name=name,
            asset_type=asset_type,
            adjust_type=adjust_type,
            short_window=best_short,
            long_window=best_long,
            output_path=cum_path,
            long_only=long_only,
            threshold=threshold,
        )

        # 4b. 3D 参数曲面图
        surf_path = (
            output_dir
            / f"{asset_type}_{symbol}_{name}_{adjust_type}_surface_{mode_suffix}{thr_suffix}.png"
        )
        draw_window_sweep_surface(
            grid,
            symbol=symbol,
            name=name,
            asset_type=asset_type,
            adjust_type=adjust_type,
            long_only=long_only,
            output_path=surf_path,
            best_short=best_short,
            best_long=best_long,
        )

        edge = best_strategy_ret - best_result.final_market_return
        summary_rows.append({
            "symbol": symbol,
            "name": name,
            "best_short": best_short,
            "best_long": best_long,
            "strategy_return": round(best_strategy_ret, 4),
            "market_return": round(best_result.final_market_return, 4),
            "edge": round(edge, 4),
        })
        print(
            f"  best=({best_short}, {best_long}) "
            f"strategy {best_strategy_ret * 100:+.2f}% / "
            f"market {best_result.final_market_return * 100:+.2f}% / "
            f"edge {edge * 100:+.2f}pp"
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        csv_path = output_dir / f"summary_{asset_type}_{adjust_type}_{mode_suffix}{thr_suffix}.csv"
        summary.to_csv(csv_path, index=False)
        print(f"\nSaved summary: {csv_path}")

    return summary


# 多变体合并图：把同一只资产、不同 (long_only, threshold) 的扫描最优结果叠加
def draw_sweep_combined(
    variant_results: list[tuple[bool, float, int, int, BacktestResult]],
    symbol: str,
    name: str,
    asset_type: AssetType,
    adjust_type: AdjustType,
    output_path: Path,
) -> None:
    """同一只资产、多个策略变体（每个用各自最优参数）叠加在一张图里。

    每条 variant 用 backtest_dual_ma 跑出来的累计净值曲线（覆盖完整时间线，起点 1.0）。
    买入持有作为 base 取自第一个 variant（所有 variant 共享同一条买入持有曲线）。
    """
    if not variant_results:
        raise ValueError("variant_results is empty")

    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(13, 7))

    # 1. 买入持有 base，所有 variant 共享，取第一个
    _, _, _, _, first_result = variant_results[0]
    market_cum = first_result.data["cumulative_market_return"]
    ax.plot(
        market_cum.index,
        market_cum.values,
        label=f"买入持有 base ({first_result.final_market_return * 100:+.1f}%)",
        color="black",
        linewidth=2.0,
        zorder=2,
    )

    # 2. 每个 variant 用最优参数下的策略累计净值曲线
    palette = ["green", "steelblue", "darkorange", "purple", "brown", "magenta"]
    for i, (long_only, threshold, best_short, best_long, result) in enumerate(variant_results):
        mode_label = "仅做多" if long_only else "多空双向"
        thr_label = f" + 阈值 {threshold * 100:.1f}%" if threshold > 0 else ""
        legend = (
            f"{mode_label}{thr_label} "
            f"({best_short}/{best_long}, {result.final_strategy_return * 100:+.1f}%)"
        )
        strat_cum = result.data["cumulative_strategy_return"]
        ax.plot(
            strat_cum.index,
            strat_cum.values,
            label=legend,
            color=palette[i % len(palette)],
            linewidth=1.5,
            zorder=3,
        )

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title(
        f"{symbol} {name} — 双均线参数扫描对照（in-sample 最优） "
        f"[{asset_type} {ADJUST_TYPE_MAP[adjust_type]}]",
        fontsize=13, fontweight="bold",
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


def calculate_and_draw_dual_ma_sweep_combined(
    combined_prices: pd.DataFrame,
    asset_type: AssetType,
    adjust_type: AdjustType,
    output_dir: Path,
    # variant 表示不同的 (long_only, threshold) 组合
    variants: list[tuple[bool, float]] | None = None,
    short_range: tuple[int, int] = (10, 50),
    long_range: tuple[int, int] = (100, 200),
    short_step: int = 1,
    long_step: int = 5,
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    """对每只资产跑多个 (long_only, threshold) 变体的参数扫描，叠加到一张图里。

    每个 variant 单独扫网格 + 找最优，然后用各自最优参数跑一次回测得到累计净值。
    单独的 3D 曲面图仍由原 calculate_and_draw_dual_ma_sweep 输出，这里只生成对照折线图。

    variants 默认 4 组：
      [(True, 0.0), (False, 0.0), (True, 0.02), (False, 0.02)]
    """
    if variants is None:
        variants = [(True, 0.0), (False, 0.0), (True, 0.02), (False, 0.02)]

    output_dir.mkdir(parents=True, exist_ok=True)
    name_to_symbol = {meta["name"]: code for code, meta in POOL[asset_type].items()}

    summary_rows: list[dict] = []
    for name in combined_prices.columns:
        prices = combined_prices[name].dropna()
        if len(prices) < long_range[1] + 2:
            print(f"[skip] {name}: too few rows ({len(prices)}) for max_long={long_range[1]}")
            continue

        symbol = name_to_symbol.get(name, name)
        print(f"[{asset_type} {symbol} {name}] sweep combined ({len(variants)} variants)...")

        variant_results: list[tuple[bool, float, int, int, "BacktestResult"]] = []
        for long_only, threshold in variants:
            grid = sweep_dual_ma_windows(
                prices,
                short_range=short_range,
                long_range=long_range,
                short_step=short_step,
                long_step=long_step,
                long_only=long_only,
                threshold=threshold,
            )
            try:
                best_short, best_long, _ = find_best_window_pair(grid)
            except ValueError as e:
                print(f"  skip ({long_only}, {threshold}): {e}")
                continue

            result = backtest_dual_ma(
                prices,
                short_window=best_short,
                long_window=best_long,
                initial_capital=initial_capital,
                long_only=long_only,
                threshold=threshold,
            )
            variant_results.append((long_only, threshold, best_short, best_long, result))
            tag = f"{'longonly' if long_only else 'longshort'}, thr={threshold:.0%}"
            print(
                f"  [{tag}] best=({best_short}, {best_long}) "
                f"strategy {result.final_strategy_return * 100:+.2f}% / "
                f"market {result.final_market_return * 100:+.2f}%"
            )

        if not variant_results:
            continue

        out_path = (
            output_dir
            / f"{asset_type}_{symbol}_{name}_{adjust_type}_sweep_combined.png"
        )
        draw_sweep_combined(
            variant_results,
            symbol=symbol,
            name=name,
            asset_type=asset_type,
            adjust_type=adjust_type,
            output_path=out_path,
        )

        for long_only, threshold, best_short, best_long, result in variant_results:
            edge = result.final_strategy_return - result.final_market_return
            summary_rows.append({
                "symbol": symbol,
                "name": name,
                "long_only": long_only,
                "threshold": threshold,
                "best_short": best_short,
                "best_long": best_long,
                "strategy_return": round(result.final_strategy_return, 4),
                "market_return": round(result.final_market_return, 4),
                "edge": round(edge, 4),
            })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        csv_path = (
            output_dir
            / f"summary_sweep_combined_{asset_type}_{adjust_type}.csv"
        )
        summary.to_csv(csv_path, index=False)
        print(f"\nSaved summary: {csv_path}")

    return summary
