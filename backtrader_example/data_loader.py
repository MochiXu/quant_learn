"""把 common.load_ohlcv 的结果包装成 backtrader 数据源.

这一层是 backtrader 专属的薄封装 ( thin wrapper ) ,
通用的 OHLCV 加载逻辑放在 common/dataloader.py 里.
"""
from __future__ import annotations

import backtrader as bt

from common.constant import POOL, AdjustType, AssetType
from common.dataloader import load_ohlcv


def make_bt_feed(
    asset_type: AssetType,
    symbol: str,
    adjust_type: AdjustType = "hfq",
    start: str | None = None,
    end: str | None = None,
) -> bt.feeds.PandasData:
    """读取标的数据并返回 backtrader 数据源 ( data feed ) .

    backtrader 内部用一个时间游标逐日推送数据给策略 ,
    这里我们用 PandasData 把 DataFrame 适配成它要的格式.
    """
    df = load_ohlcv(asset_type, symbol, adjust_type, start=start, end=end)
    name = POOL[asset_type][symbol]["name"]
    # PandasData 默认会按列名找 open / high / low / close / volume , 我们的列名正好对得上.
    # openinterest ( 持仓量 , 期货才有 ) 在 ETF / 股票场景没有 , 设为 -1 让 backtrader 忽略.
    return bt.feeds.PandasData(dataname=df, name=name, openinterest=-1)
