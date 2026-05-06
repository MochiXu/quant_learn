"""02 - Buy and Hold: 第一次下单.

学习目标:
- 用 self.buy() / self.sell() 下市价单 ( market order ) .
- 用 self.position 检查当前是否持仓.
- 用 notify_order 监听订单状态变化 ( 提交 / 成交 / 拒绝 ) .

策略:
    第一天买入 , 持有到底 ( buy and hold , 量化里最朴素的基准 baseline ) .

运行:
    python backtrader_example/02_buy_and_hold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtrader as bt

from backtrader_example.data_loader import make_bt_feed


class BuyAndHoldStrategy(bt.Strategy):
    def __init__(self) -> None:
        self.order: bt.Order | None = None  # 跟踪当前在途订单 ( pending order )

    def next(self) -> None:
        # 已经持仓就什么都不做 , 一直拿着.
        if self.position:
            return
        if self.order is not None:
            return  # 已经在排队 , 别重复下单.

        # 默认 size = 1 ( 只买 1 股 ) , 这里用账户 95% 资金估算可买多少股.
        # 留 5% 当缓冲 ( buffer ) 防止下一根 bar 开盘价上涨买不进.
        cash = self.broker.getcash()
        price = self.datas[0].close[0]
        size = int(cash * 0.95 / price)
        if size > 0:
            self.order = self.buy(size=size)
            print(f"{self._dt()}  下买单 size={size}  ref close={price:.3f}")

    def notify_order(self, order: bt.Order) -> None:
        # 订单经历 Submitted -> Accepted -> Completed ( 或 Rejected / Canceled ) .
        # 我们只关心终态 , 进行中的状态忽略.
        if order.status in (order.Submitted, order.Accepted):
            return

        if order.status == order.Completed:
            side = "买入" if order.isbuy() else "卖出"
            print(
                f"{self._dt()}  {side}成交  price={order.executed.price:.3f}  "
                f"size={order.executed.size}  cost={order.executed.value:.2f}"
            )
        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            print(f"{self._dt()}  订单失败 status={order.getstatusname()}")

        self.order = None  # 清掉跟踪句柄.

    def _dt(self) -> str:
        return self.datas[0].datetime.date(0).isoformat()


def main() -> None:
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BuyAndHoldStrategy)

    feed = make_bt_feed("fund", "510300", "hfq", start="2018-01-01", end="2024-12-31")
    cerebro.adddata(feed)

    cerebro.broker.setcash(100_000.0)

    start_value = cerebro.broker.getvalue()
    print(f"起始资金 : {start_value:.2f}")
    cerebro.run()
    end_value = cerebro.broker.getvalue()
    print(f"结束资金 : {end_value:.2f}")
    print(f"总收益率 ( total return ) : {(end_value / start_value - 1) * 100:.2f}%")

    cerebro.plot(style="candle")  # 画 K 线图 + 资金曲线 ( equity curve ) .


if __name__ == "__main__":
    main()
