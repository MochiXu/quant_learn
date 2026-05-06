"""06 - Optimization: 双均线参数寻优 ( parameter optimization ) .

学习目标:
- 用 cerebro.optstrategy () 替代 addstrategy , 让 backtrader 在参数网格 ( grid ) 上跑遍历.
- 多进程并行 ( backtrader 内部用 multiprocessing ) , 注意 Mac / Windows 必须用
  if __name__ == "__main__" 守卫 ( guard ) , 否则会无限 fork.
- 收集每组参数的指标 , 排序找最优.

警告 : 参数寻优很容易过拟合 ( overfitting ) ,
找出来的 "最优" 可能只是历史巧合 , 出样本就失效.
正经做法是用样本外检验 ( out-of-sample test ) 或 walk-forward 验证 ,
本例只演示语法 .

运行:
    python backtrader_example/06_optimization.py
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
                self.order = self.buy()
        else:
            if self.crossover[0] < 0:
                self.order = self.close()

    def notify_order(self, order: bt.Order) -> None:
        if order.status not in (order.Submitted, order.Accepted):
            self.order = None


def main() -> None:
    cerebro = bt.Cerebro(optreturn=True)  # optreturn = 只返回参数 + analyzers , 节约内存.

    feed = make_bt_feed("fund", "510300", "hfq", start="2018-01-01", end="2024-12-31")
    cerebro.adddata(feed)

    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0003)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        annualize=True,
        riskfreerate=0.02,
    )
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

    # 参数网格 ( grid ) :
    #   fast ∈ {5, 10, 15, 20}
    #   slow ∈ {30, 50, 80, 120}
    # 共 4 x 4 = 16 组组合.
    # 注意 : 运行时 backtrader 会自动跳过 fast >= slow 的无意义组合吗 ? 不会 , 我们也不去过滤 ,
    # 让结果里这些组合自然表现差 , 顺便观察一下 .
    cerebro.optstrategy(
        SmaCrossStrategy,
        fast=[5, 10, 15, 20],
        slow=[30, 50, 80, 120],
    )

    print("开始参数寻优 ( 这会跑 16 次回测 , 稍等 ) ...")
    results = cerebro.run()  # results : list of list of strategy instances.

    rows: list[tuple[int, int, float, float, float, float]] = []
    for run in results:
        strat = run[0]
        fast = strat.params.fast
        slow = strat.params.slow
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio") or float("nan")
        ret = strat.analyzers.returns.get_analysis().get("rnorm100", 0.0)
        dd = strat.analyzers.dd.get_analysis().max.drawdown
        total = strat.analyzers.returns.get_analysis().get("rtot", 0.0) * 100
        rows.append((fast, slow, sharpe, ret, dd, total))

    # 按夏普降序排序 ( NaN 放最后 ) .
    rows.sort(key=lambda r: (r[2] != r[2], -r[2] if r[2] == r[2] else 0))

    print("\n=== 参数寻优结果 ( 按年化夏普降序 ) ===")
    print(f"{'fast':>5}  {'slow':>5}  {'Sharpe':>8}  {'AnnRet%':>8}  {'MaxDD%':>8}  {'TotRet%':>8}")
    for fast, slow, sharpe, ret, dd, total in rows:
        print(f"{fast:>5}  {slow:>5}  {sharpe:>8.3f}  {ret:>8.2f}  {dd:>8.2f}  {total:>8.2f}")

    best = rows[0]
    print(f"\n最优参数 : fast={best[0]}, slow={best[1]}, Sharpe={best[2]:.3f}")
    print("提示 : 这个 ' 最优 ' 仅在 2018-2024 历史区间成立 , 别盲信.")


if __name__ == "__main__":
    main()
