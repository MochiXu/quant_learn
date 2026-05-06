"""05 - Analyzers: 用内置分析器算夏普 / 回撤 / 年化收益.

学习目标:
- Analyzer ( 分析器 ) 是 backtrader 在策略外挂载的统计组件 ,
  跑完回测后能拿到结构化结果 , 不用自己写 .
- 对照你之前用 pandas 手算的版本 , 验证结果是否一致.

本文件用到的 Analyzer:
    SharpeRatio  - 夏普比率 ( Sharpe ratio , 风险调整收益 ) .
    DrawDown     - 回撤 ( drawdown ) , 包括最大回撤 ( max drawdown ) 和持续时间.
    Returns      - 总收益 / 年化收益 ( annualized return ) .
    TradeAnalyzer - 交易级统计 ( 胜率 / 平均盈亏等 ) .

运行:
    python backtrader_example/05_analyzers.py
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
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCrossStrategy)

    feed = make_bt_feed("fund", "510300", "hfq", start="2018-01-01", end="2024-12-31")
    cerebro.adddata(feed)

    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0003)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    # 添加分析器 . _name 是回测结束后取结果用的 key.
    # timeframe = Days + annualize = True : 让 Sharpe 按日频算并年化 ( 252 个交易日 ) .
    # riskfreerate : 无风险利率 ( risk-free rate ) , 这里设 2% , 相当于货币基金回报 .
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        annualize=True,
        riskfreerate=0.02,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    print(f"起始资金 : {cerebro.broker.getvalue():.2f}")
    # cerebro.run () 返回一个策略实例列表 ( 多策略时每个一项 ) .
    results = cerebro.run()
    strat = results[0]
    end_value = cerebro.broker.getvalue()
    print(f"结束资金 : {end_value:.2f}")

    # 取分析器结果.
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    print("\n=== 业绩指标 ( performance metrics ) ===")
    print(f"年化夏普 ( annualized Sharpe ) : {sharpe.get('sharperatio')}")
    print(f"总收益 ( total return ) : {returns.get('rtot', 0) * 100:.2f}%")
    print(f"年化收益 ( annualized return ) : {returns.get('rnorm100', 0):.2f}%")
    print(f"最大回撤 ( max drawdown ) : {dd.max.drawdown:.2f}%")
    print(f"最大回撤持续 ( max DD length ) : {dd.max.len} 个交易日")

    print("\n=== 交易统计 ===")
    total = trades.get("total", {})
    won = trades.get("won", {})
    lost = trades.get("lost", {})
    total_closed = total.get("closed", 0)
    won_total = won.get("total", 0)
    print(f"总交易笔数 : {total_closed}")
    if total_closed > 0:
        print(f"胜率 ( win rate ) : {won_total / total_closed * 100:.1f}%")
    print(f"盈利总额 : {won.get('pnl', {}).get('total', 0):.2f}")
    print(f"亏损总额 : {lost.get('pnl', {}).get('total', 0):.2f}")

    cerebro.plot(style="candle")


if __name__ == "__main__":
    main()
