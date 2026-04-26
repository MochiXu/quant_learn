from __future__ import annotations

from pathlib import Path

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common.draw import configure_chinese_font
from common.constant import ADJUST_TYPE_MAP, AssetType, AdjustType


# 计算并绘制风险收益散点图（夏普比率）
# X 轴 = 风险（年化波动率），Y 轴 = 收益（年化收益率）
# 点越大/颜色越亮 = 夏普比率越高 = 性价比越好
def calculate_and_draw_sharpe_ratio(
    combined_prices: pd.DataFrame,
    asset_type: AssetType,
    adjust_type: AdjustType,
    output_path: Path,
) -> None:
    configure_chinese_font()

    # 计算每日收益率
    returns = combined_prices.pct_change(fill_method=None).dropna()
    # 年化收益率 = 日均收益率 × 252（一年约 252 个交易日）
    annual_returns = returns.mean() * 252
    # 年化波动率 = 日标准差 × √252
    annual_volatility = returns.std() * np.sqrt(252)

    # 夏普比率 = (年化收益 - 无风险利率) / 年化波动率
    risk_free_rate = 0.02
    sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility

    risk_return_metrics = pd.DataFrame({
        'Return': annual_returns,
        'Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
    })

    print(risk_return_metrics.sort_values('Sharpe Ratio', ascending=False))

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=risk_return_metrics,
        x='Volatility',
        y='Return',
        size='Sharpe Ratio',
        sizes=(50, 500),
        legend='brief',
        hue='Sharpe Ratio',
        palette='viridis',
    )

    for i, asset in enumerate(risk_return_metrics.index):
        plt.annotate(
            asset,
            (risk_return_metrics['Volatility'].iloc[i], risk_return_metrics['Return'].iloc[i]),
            xytext=(5, 5),
            textcoords='offset points',
        )

    plt.title(f'Sharpe Ratio ({asset_type} {adjust_type} {ADJUST_TYPE_MAP[adjust_type]})')
    plt.xlabel('Risk (Annualized Volatility)')
    plt.ylabel('Return (Annualized)')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
