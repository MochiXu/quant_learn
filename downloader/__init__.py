import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downloader.stock import StockDownloader, StockDownloadParam
from downloader.fund import FundDownloader, FundDownloadParam
from common.constant import POOL, SECTOR_ETFS

__all__ = [
    "StockDownloader",
    "StockDownloadParam",
    "FundDownloader",
    "FundDownloadParam",
    "POOL",
    "SECTOR_ETFS",
]


def main() -> None:
    # Funds: POOL["fund"] + sector-rotation ETFs (deduplicated, order preserved).
    fund_symbols = list(dict.fromkeys(
        list(POOL["fund"].keys()) + list(SECTOR_ETFS.keys())
    ))
    fund_param = FundDownloadParam(symbols=fund_symbols)
    FundDownloader(fund_param).download_all()

    stock_param = StockDownloadParam(
        symbols=list(POOL["stock"].keys()),
    )
    StockDownloader(stock_param).download_all()


if __name__ == "__main__":
    main()
