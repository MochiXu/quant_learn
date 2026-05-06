"""04 - Sizer & Commission: 让回测更接近真实.

学习目标:
- 用 cerebro.broker.setcommission () 设置手续费 ( commission ) , 默认是 0.
- 用 Sizer ( 仓位管理器 , position sizer ) 决定每次下单买多少 ,
  默认 backtrader 一次只买 1 股 , 这显然不是我们想要的.
- 对比有 / 无费用对最终收益的影响.

策略沿用 03 的双均线 , 不重复造轮子.

A 股 ETF 真实费用参考 ( 2026 年估算 ) :
    - 券商佣金 ( commission ) : 万一到万三 ( 0.01% ~ 0.03% ) , 双边收取.
    - 印花税 ( stamp duty ) : ETF 免征.
    - 这里取万三 = 0.0003.

运行:
    python backtrader_example/04_sizer_commission.py
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
    """和 03 几乎一样 , 只是这里不再手动算 size ( 交给 Sizer ) ."""

    params = dict(fast=10, slow=30)

    def __init__(self) -> None:
        close = self.datas[0].close
        self.sma_fast = bt.indicators.SMA(close, period=self.p.fast)
        self.sma_slow = bt.indicators.SMA(close, period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
        self.order: bt.Order | None = None

    def next(self) -> None:
        if self.order is not None:
            return
        if not self.position:
            if self.crossover[0] > 0:
                # 注意 : 不再传 size , 由 Sizer 决定.
                self.order = self.buy()
        else:
            if self.crossover[0] < 0:
                self.order = self.close()

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status == order.Completed:
            side = "买入" if order.isbuy() else "卖出"
            # commission = 该笔订单的手续费.
            print(
                f"{self._dt()}  {side}  price={order.executed.price:.3f}  "
                f"size={order.executed.size}  comm={order.executed.comm:.2f}"
            )
        self.order = None

    def _dt(self) -> str:
        return self.datas[0].datetime.date(0).isoformat()


def run_one(commission: float, label: str) -> float:
    """跑一次回测 , 返回最终账户价值. label 仅用于日志."""
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCrossStrategy)

    feed = make_bt_feed("fund", "510300", "hfq", start="2018-01-01", end="2024-12-31")
    cerebro.adddata(feed)

    cerebro.broker.setcash(100_000.0)
    # 手续费按成交金额比例收取 ( commtype = CommInfoBase.COMM_PERC , 默认就是百分比模式 ) .
    cerebro.broker.setcommission(commission=commission)

    # PercentSizer : 每次下单使用账户资金的 X%.
    # 这里用 95% , 留 5% 作为手续费缓冲 + 价格波动缓冲.
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    print(f"\n=== {label} ( commission={commission} ) ===")
    print(f"起始资金 : {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    end_value = cerebro.broker.getvalue()
    print(f"结束资金 : {end_value:.2f}")
    return end_value


def main() -> None:
    # 对比 : 0 费用 vs 万三费用.
    no_fee = run_one(commission=0.0, label="无手续费")
    with_fee = run_one(commission=0.0003, label="万三手续费")

    print("\n=== 对比 ===")
    print(f"无费用结束资金   : {no_fee:.2f}")
    print(f"万三结束资金     : {with_fee:.2f}")
    print(f"费用拖累 ( drag ) : {no_fee - with_fee:.2f}")


if __name__ == "__main__":
    main()
