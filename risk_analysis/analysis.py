from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from risk_analysis.efficient_fontier import calculate_and_draw_efficient_frontier
from risk_analysis.sharpe_ratio import calculate_and_draw_sharpe_ratio
from risk_analysis.correlation_heatmap import calculate_and_draw_correlation_heatmap
from risk_analysis.cumulative_returns import calculate_cumulative_returns, draw_cumulative_returns


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


if __name__ == "__main__":
    main()
