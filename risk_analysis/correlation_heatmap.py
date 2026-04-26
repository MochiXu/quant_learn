from __future__ import annotations

import sys
from pathlib import Path

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from common.constant import ADJUST_TYPE_MAP, AssetType, AdjustType


# 计算并绘制资产收益率相关性热力图
# - returns: 各资产每日收益率
def calculate_and_draw_correlation_heatmap(
    combined_prices: pd.DataFrame,
    asset_type: AssetType,
    adjust_type: AdjustType,
    output_path: Path
) -> None:
    """Calculate and draw correlation heatmap for the returns DataFrame."""
    returns = combined_prices.pct_change(fill_method=None).dropna()
    correlation_matrix = returns.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Correlation Heatmap ({asset_type} {adjust_type} {ADJUST_TYPE_MAP[adjust_type]})')
    plt.savefig(output_path)
    plt.close()


