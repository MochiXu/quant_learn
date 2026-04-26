from __future__ import annotations

from pathlib import Path

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common.draw import configure_chinese_font
from common.constant import ADJUST_TYPE_MAP, AssetType, AdjustType
from scipy.optimize import minimize

# 计算单个组合的 年化收益率 和 年化波动率
def portfolio_performance(weights: np.ndarray, returns: pd.DataFrame) -> tuple[float, float]:
    # 年化收益率: 每个资产的收益率 * 权重, SUM 之后 * 252
    portfolio_return = np.sum(returns.mean() * weights) * 252
    # 年化波动率: 权重 * 协方差矩阵 * 权重, 再开平方
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float) -> float:
    portfolio_return, portfolio_volatility = portfolio_performance(weights, returns)
    return -(portfolio_return - risk_free_rate) / portfolio_volatility


# 生成 1000 个随机组合的夏普比率
def random_portfolio_efficient_frontier(
    returns: pd.DataFrame,
    risk_free_rate: float,
    num_portfolios: int = 10000
) -> pd.DataFrame:
    num_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights) # normalize weights
        p_return, p_volatility = portfolio_performance(weights, returns)
        results[0, i] = p_return
        results[1, i] = p_volatility
        results[2, i] = p_return / p_volatility # sharpe ratio
    return results.T

def optimize_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: float
) -> tuple[np.ndarray, float]:
    # 资产数量
    num_assets = len(returns.columns)
    # Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # 权重范围 (0, 1)
    bounds = tuple((0, 1) for _ in range(num_assets))
    # 梯度优化
    result = minimize(
        negative_sharpe_ratio,
        num_assets * [1. / num_assets],
        args=(returns, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    optimal_weights = result.x
    optimal_return, optimal_volatility = portfolio_performance(optimal_weights, returns)
    return optimal_weights, optimal_return, optimal_volatility


def calculate_and_draw_efficient_frontier(
    combined_prices: pd.DataFrame,
    output_path: Path,
    asset_type: AssetType,
    adjust_type: AdjustType,
    risk_free_rate: float = 0.02,
    num_portfolios: int = 10000,
) -> None:
    configure_chinese_font()

    returns = combined_prices.pct_change(fill_method=None).dropna()
    efficient_frontier = random_portfolio_efficient_frontier(returns, risk_free_rate, num_portfolios)
    optimal_weights, optimal_return, optimal_volatility = optimize_portfolio(returns, risk_free_rate)

    # 左边画散点图，右边画权重表
    fig, (ax_scatter, ax_table) = plt.subplots(
        1, 2, figsize=(20, 10),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # 散点图
    scatter = ax_scatter.scatter(
        efficient_frontier[:, 1],
        efficient_frontier[:, 0],
        c=efficient_frontier[:, 2],
        cmap='viridis',
        alpha=0.6,
        s=10,
    )
    fig.colorbar(scatter, ax=ax_scatter, label='Sharpe Ratio')
    ax_scatter.scatter(
        optimal_volatility, optimal_return,
        c='red', s=300, marker='*', zorder=5, label='Optimal Portfolio',
    )
    ax_scatter.set_title(f'Efficient Frontier ({asset_type} {adjust_type} {ADJUST_TYPE_MAP[adjust_type]})', fontsize=14)
    ax_scatter.set_xlabel('Risk (Annualized Volatility)')
    ax_scatter.set_ylabel('Return (Annualized)')
    ax_scatter.legend(fontsize=10)

    # 权重表
    ax_table.axis('off')
    ax_table.set_title('Optimal Weights', fontsize=14, pad=20)

    table_data = [[asset, f"{weight:.2%}"] for asset, weight in zip(returns.columns, optimal_weights)]
    table_data.append(['Expected Return', f"{optimal_return:.2%}"])
    table_data.append(['Expected Volatility', f"{optimal_volatility:.2%}"])
    sharpe = (optimal_return - risk_free_rate) / optimal_volatility
    table_data.append(['Sharpe Ratio', f"{sharpe:.3f}"])

    table = ax_table.table(
        cellText=table_data,
        colLabels=['Asset', 'Value'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")