import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downloader.stock import StockDownloader, StockDownloadParam
from downloader.fund import FundDownloader, FundDownloadParam
from constant import POOL

__all__ = [
    "StockDownloader",
    "StockDownloadParam",
    "FundDownloader",
    "FundDownloadParam",
    "POOL",
]


def main() -> None:
    fund_param = FundDownloadParam(
        symbols=list(POOL["fund"].keys()),
    )
    FundDownloader(fund_param).download_all()

    stock_param = StockDownloadParam(
        symbols=list(POOL["stock"].keys()),
    )
    StockDownloader(stock_param).download_all()


if __name__ == "__main__":
    main()
