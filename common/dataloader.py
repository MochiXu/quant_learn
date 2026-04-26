

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