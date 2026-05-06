

from common.constant import AssetType
import pandas as pd
from common.constant import FUND_DIR, STOCK_DIR, POOL, AdjustType


def load_close_prices(asset_type: AssetType, adjust_type: AdjustType) -> pd.DataFrame:
    """Return a wide DataFrame of close prices, one column per symbol."""
    if asset_type == "fund":
        base_dir = FUND_DIR / adjust_type
    elif asset_type == "stock":
        base_dir = STOCK_DIR / adjust_type
    else:
        raise ValueError(f"Unknown asset_type: {asset_type}")

    series_list: list[pd.Series] = []
    for symbol in POOL[asset_type].keys():
        path = base_dir / f"{symbol}.parquet"
        name = POOL[asset_type][symbol]['name']
        if not path.exists():
            print(f"[skip] missing {asset_type} {symbol}: {path}")
            continue

        print(f"reading name:'{name}', path:'{path}'")

        df = pd.read_parquet(path, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        # 当前 df 的 index 是一个默认列, 此处需要将 date 设置为 index 列
        df = df.set_index("date")
        # 获取 close 列，并按照 date 列排序
        close_series = df["close"].sort_index()
        # 去掉重复的日期，保留第一条
        close_series = close_series[~close_series.index.duplicated(keep='first')]
        # 将 name 从 close price 修改为对应的 symbol
        # close_series.name = symbol
        close_series.name = name
        print(close_series.head())

        series_list.append(close_series)
        print("-----")

    if not series_list:
        raise RuntimeError(f"No data loaded for asset_type={asset_type}")

    return pd.concat(series_list, axis=1)

def load_all_close_prices(adjust_type: AdjustType) -> pd.DataFrame:
    """Return a df of close prices, one column per symbol."""
    return pd.concat([load_close_prices("fund", adjust_type), load_close_prices("stock", adjust_type)], axis=1)


def load_ohlcv(
    asset_type: AssetType,
    symbol: str,
    adjust_type: AdjustType,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """加载单个标的的 OHLCV ( open / high / low / close / volume ) 数据.

    返回的 DataFrame 以 date 为 DatetimeIndex , 列固定为 open / high / low / close / volume ,
    适合直接喂给 backtrader 的 PandasData ( 一个把 pandas DataFrame 转成 backtrader 数据源的类 ) .

    Args:
        asset_type: "fund" 或 "stock" .
        symbol: 标的代码 , 例如 "510300" .
        adjust_type: "raw" / "qfq" / "hfq" , 复权类型 ( adjustment type ) .
        start: 起始日期 , 格式 "YYYY-MM-DD" , None 表示不限.
        end: 结束日期 , 格式 "YYYY-MM-DD" , None 表示不限.
    """
    if asset_type == "fund":
        base_dir = FUND_DIR / adjust_type
    elif asset_type == "stock":
        base_dir = STOCK_DIR / adjust_type
    else:
        raise ValueError(f"Unknown asset_type: {asset_type}")

    path = base_dir / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {asset_type} {symbol}: {path}")

    cols = ["date", "open", "high", "low", "close", "volume"]
    df = pd.read_parquet(path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # 去重 ( 保留首个 ) , 防止源数据里同一天出现多行
    df = df[~df.index.duplicated(keep="first")]
    # backtrader 对 NaN 敏感 ( 会让指标算出怪值 ) , 这里直接丢掉缺失行
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    if start is not None:
        df = df.loc[pd.to_datetime(start):]
    if end is not None:
        df = df.loc[:pd.to_datetime(end)]

    return df