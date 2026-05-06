from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from risk_analysis.efficient_fontier import calculate_and_draw_efficient_frontier
from risk_analysis.sharpe_ratio import calculate_and_draw_sharpe_ratio
from risk_analysis.correlation_heatmap import calculate_and_draw_correlation_heatmap
from risk_analysis.cumulative_returns import calculate_cumulative_returns, draw_cumulative_returns
from risk_analysis.backtest_dual_ma import calculate_and_draw_dual_ma_backtest
from risk_analysis.backtest_dual_ma_sweep import (
    calculate_and_draw_dual_ma_sweep,
    calculate_and_draw_dual_ma_sweep_combined,
)
from risk_analysis.backtest_dual_ma_walk_forward import (
    calculate_and_draw_dual_ma_walk_forward,
    calculate_and_draw_dual_ma_walk_forward_combined,
)


from common.dataloader import load_all_close_prices, load_close_prices
from common.constant import AdjustType

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "risk_analysis"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adjust_type: AdjustType = "hfq"

    for asset_type in ("fund", "stock"):
        # 获取所有资产的收盘价
        combined_prices = load_close_prices(asset_type, adjust_type)
        # all_combined_prices = load_all_close_prices(adjust_type)

        # 计算并绘制资产之间的收益率相关性热力图
        calculate_and_draw_correlation_heatmap(
            combined_prices,
            asset_type,
            adjust_type,
            OUTPUT_DIR / f"correlation_heatmap_{asset_type}_{adjust_type}.png",
        )

        # 计算各资产的累计收益率
        cumulative_returns = calculate_cumulative_returns(combined_prices)

        # 绘制各资产的累计收益率图
        draw_cumulative_returns(
            cumulative_returns,
            asset_type,
            adjust_type,
            OUTPUT_DIR / f"cumulative_returns_{asset_type}_{adjust_type}.png",
        )

        # 计算并绘制夏普比率图
        calculate_and_draw_sharpe_ratio(
            combined_prices,
            asset_type,
            adjust_type,
            OUTPUT_DIR / f"sharpe_ratio_{asset_type}_{adjust_type}.png",
        )

        # 绘制 efficient_fontier
        calculate_and_draw_efficient_frontier(
            combined_prices,
            OUTPUT_DIR / f"efficient_frontier_{asset_type}_{adjust_type}.png",
            asset_type,
            adjust_type,
        )

        # 双均线动量策略回测：每只资产单独出一张图
        # 第三个参数是输出子文件夹名，因为按 asset_type × symbol 会生成很多张图，集中放在一个子目录里
        # calculate_and_draw_dual_ma_backtest(
        #     combined_prices,
        #     asset_type,
        #     adjust_type,
        #     OUTPUT_DIR / "backtest",
        #     short_window=20,
        #     long_window=50,
        # )

        # 双均线参数扫描合并图：每只资产把 4 个变体（仅做多 / 多空双向 ± 2% 阈值）
        # 各自找最优参数后的累计净值曲线叠加在同一张图里，跟买入持有 base 对比。
        # 4 个变体共享一个 universe：[(True, 0.0), (False, 0.0), (True, 0.02), (False, 0.02)]
        calculate_and_draw_dual_ma_sweep_combined(
            combined_prices,
            asset_type,
            adjust_type,
            OUTPUT_DIR / "backtest_sweep",
            short_range=(5, 60),
            long_range=(100, 300),
            short_step=1,
            long_step=5,
        )

        # 双均线 walk-forward 合并图：同样 4 个变体叠加，每只资产一张图。
        # 包含完整时间线的买入持有 base + 各变体的 OOS 累计净值，OOS 起点用红色竖线标注。
        calculate_and_draw_dual_ma_walk_forward_combined(
            combined_prices,
            asset_type,
            adjust_type,
            OUTPUT_DIR / "backtest_walk_forward",
            short_range=(5, 60),
            long_range=(100, 300),
            short_step=1,
            long_step=5,
            train_years=3,
            test_years=1,
        )


if __name__ == "__main__":
    main()
