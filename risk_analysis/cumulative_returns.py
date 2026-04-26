from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from common.draw import configure_chinese_font
from common.constant import ADJUST_TYPE_MAP, POOL, AssetType, AdjustType



# 计算每个资产的累计收益率，从 1.0 出发
# 公式：cumulative = (1 + 日收益率).cumprod()
# 最终值 1.15 表示总共涨了 15%，0.90 表示总共跌了 10%
def calculate_cumulative_returns(combined_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute (1 + r).cumprod() per asset, each starting at 1.0 on its first date."""
    cum = pd.DataFrame(index=combined_prices.index, columns=combined_prices.columns, dtype=float)
    for col in combined_prices.columns:
        # 取出单个资产的收盘价，去掉缺失值
        series = combined_prices[col].dropna()
        if len(series) < 2:
            continue
        # 计算日收益率：(今天 - 昨天) / 昨天，第一天没有前一天所以 dropna
        returns = series.pct_change(fill_method=None).dropna()
        # 转成乘数后逐日累乘，例如 [0.9896, 1.0316] -> cumprod -> [0.9896, 1.0209]
        cum_col = (1 + returns).cumprod()
        print("cum_col.head()---\n", cum_col.head())
        # 在第一个价格日期补上 1.0 作为起点（因为 pct_change 会丢掉第一天）
        start = pd.Series([1.0], index=[series.index[0]])
        cum_col = pd.concat([start, cum_col])
        # 重写 cum 表中对应的列
        cum.loc[cum_col.index, col] = cum_col
    return cum


def _legend_label(symbol: str) -> str:
    # iter all asset_type and all symbols, return the name of the symbol
    for asset_type in POOL.keys():
        meta = POOL[asset_type].get(symbol, {})
        name = meta.get("name", "")
        if name:
            return f"{symbol} {name}"
    return symbol


def draw_cumulative_returns(
    cumulative_returns: pd.DataFrame,
    asset_type: AssetType,
    adjust_type: AdjustType,
    output_path: Path,
) -> None:
    configure_chinese_font()

    fig, ax = plt.subplots(figsize=(13, 7))
    for col in cumulative_returns.columns:
        ax.plot(
            cumulative_returns.index,
            cumulative_returns[col],
            label=_legend_label(col),
            linewidth=1.5
        )

    title_map = {
        "fund": f"Fund Cumulative Returns ({adjust_type} {ADJUST_TYPE_MAP[adjust_type]})",
        "stock": f"Stock Cumulative Returns ({adjust_type} {ADJUST_TYPE_MAP[adjust_type]})",
    }
    ax.set_title(title_map[asset_type], fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns, starting at 1.0")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
