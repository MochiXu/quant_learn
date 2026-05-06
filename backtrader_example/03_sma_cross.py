"""03 - SMA Cross: 双均线策略 , 对照 pandas 手算版本.

学习目标:
- 用 backtrader 内置指标 ( indicator ) bt.indicators.SMA , 不用再手写 rolling.mean.
- 用 bt.indicators.CrossOver 检测金叉 / 死叉 ( golden cross / death cross ) .
- 在 next 里读取指标值 , 触发买卖.

策略:
    短期均线 ( fast SMA , 默认 10 日 ) 上穿长期均线 ( slow SMA , 默认 30 日 ) -> 满仓买入.
    短期下穿长期 -> 清仓卖出.

运行:
    python backtrader_example/03_sma_cross.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtrader as bt

from backtrader_example.data_loader import make_bt_feed


class SmaCrossStrategy(bt.Strategy):
    # params 是 backtrader 的标准参数声明方式 , 外部可以通过 addstrategy 传入覆盖.
    params = dict(fast=10, slow=30)

    def __init__(self) -> None:
        close = self.datas[0].close
        # SMA = Simple Moving Average ( 简单移动平均 ) .
        self.sma_fast = bt.indicators.SMA(close, period=self.p.fast)
        self.sma_slow = bt.indicators.SMA(close, period=self.p.slow)
        # CrossOver 的取值 : +1 表示 fast 上穿 slow ( 金叉 ) , -1 表示下穿 ( 死叉 ) , 0 无事件.
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
        self.order: bt.Order | None = None

    def next(self) -> None:
        if self.order is not None:
            return  # 有在途订单时不重复下单.

        if not self.position:
            # 空仓 + 金叉 -> 买入.
            if self.crossover[0] > 0:
                cash = self.broker.getcash()
                price = self.datas[0].close[0]
                size = int(cash * 0.95 / price)
                if size > 0:
                    self.order = self.buy(size=size)
                    print(f"{self._dt()}  金叉买入 size={size}  close={price:.3f}")
        else:
            # 持仓 + 死叉 -> 清仓.
            if self.crossover[0] < 0:
                self.order = self.close()  # close () 是 backtrader 的清仓快捷方法.
                print(f"{self._dt()}  死叉清仓")

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status == order.Completed:
            side = "买入" if order.isbuy() else "卖出"
            print(
                f"{self._dt()}  {side}成交  price={order.executed.price:.3f}  "
                f"size={order.executed.size}"
            )
        self.order = None

    def _dt(self) -> str:
        return self.datas[0].datetime.date(0).isoformat()


def main() -> None:
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCrossStrategy, fast=10, slow=30)

    feed = make_bt_feed("fund", "510300", "hfq", start="2018-01-01", end="2024-12-31")
    cerebro.adddata(feed)

    cerebro.broker.setcash(100_000.0)

    start_value = cerebro.broker.getvalue()
    print(f"起始资金 : {start_value:.2f}")
    cerebro.run()
    end_value = cerebro.broker.getvalue()
    print(f"结束资金 : {end_value:.2f}")
    print(f"总收益率 : {(end_value / start_value - 1) * 100:.2f}%")

    cerebro.plot(style="candle")


if __name__ == "__main__":
    main()
