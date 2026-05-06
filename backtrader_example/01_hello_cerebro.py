"""01 - Hello Cerebro: 跑通最小回测骨架.

学习目标:
- 认识 backtrader 的核心引擎 Cerebro ( 西班牙语 "大脑" , 负责协调数据 / 策略 / 经纪商 ) .
- 用 PandasData 加载我们已有的 ETF parquet 数据.
- 写一个最简策略 , 每天打印一次收盘价.

运行:
    python backtrader_example/01_hello_cerebro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让脚本可以独立运行 ( 直接 python xxx.py ) , 把项目根目录加到 sys.path.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtrader as bt

from backtrader_example.data_loader import make_bt_feed


class PrintCloseStrategy(bt.Strategy):
    """每个 bar ( K 线 ) 触发一次 next , 打印当天收盘价."""

    def next(self) -> None:
        # self.datas[0] 是我们 adddata 进去的第一个数据源.
        # self.datas[0].close[0] 表示当前 bar 的收盘价 ( [0] 是当前 , [-1] 是前一天 ) .
        dt = self.datas[0].datetime.date(0)
        close = self.datas[0].close[0]
        print(f"{dt.isoformat()}  close = {close:.3f}")


def main() -> None:
    cerebro = bt.Cerebro()
    cerebro.addstrategy(PrintCloseStrategy)

    # 默认标的 : 510300 沪深 300ETF , 后复权 ( hfq ) .
    # 时间范围故意取很短 , 避免刷屏.
    feed = make_bt_feed("fund", "510300", "hfq", start="2024-01-01", end="2024-01-31")
    cerebro.adddata(feed)

    cerebro.broker.setcash(100_000.0)

    print(f"起始资金 ( starting cash ) : {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"结束资金 ( ending cash ) : {cerebro.broker.getvalue():.2f}")
    # 这一步没下任何单 , 所以期末资金等于起始资金.


if __name__ == "__main__":
    main()
