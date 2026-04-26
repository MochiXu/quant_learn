# 量化交易学习笔记 · 场景分析与压力测试（Scenario Analysis & Stress Testing）

---

## 一、这节在做什么？

前面的 VaR/CVaR 基于**历史数据**衡量风险，场景分析则是**主动假设未来**会发生什么：
给每个资产的历史收益率叠加一个"冲击值（shock）"，重新计算组合指标，对比各场景下的表现。

核心问题：**如果市场发生某种极端情况，我的组合会受到多大冲击？**

---

## 二、五种场景定义

```python
scenarios = {
    'Base Case':             {asset: 0    for asset in returns.columns},
    'Market Crash':          {asset: -0.3 for asset in returns.columns},
    'Economic Boom':         {asset: 0.2  for asset in returns.columns},
    'Rising Interest Rates': {'US Aggregate Bonds': -0.1, 'US Treasury Bonds': -0.15},
    'Commodity Boom':        {'Gold': 0.25, 'Commodities': 0.3}
}
```

| 场景 | 背景含义 | 冲击对象 |
|------|----------|----------|
| Base Case | 基准，不施加任何冲击 | 无 |
| Market Crash | 系统性崩盘（如2008金融危机、2020疫情） | 全部资产 -30% |
| Economic Boom | 经济高速增长，全面上涨 | 全部资产 +20% |
| Rising Interest Rates | 美联储加息，债券价格下跌（利率↑ → 价格↓） | 仅债券 |
| Commodity Boom | 大宗商品超级周期，石油/黄金暴涨 | 仅商品/黄金 |

---

## 三、冲击如何施加：`shocked_returns = returns + shock_series`

`returns` 是形状为 `(N, 9)` 的 DataFrame，N 为交易日数，9 列为各资产日收益率。
`shock_series` 是由 shock 字典转换来的 pandas Series。

pandas 在 DataFrame 和 Series 相加时，会**按列名自动对齐**，再将 shock 值广播到该列所有 N 行。

**示例一：Market Crash（所有资产都有 shock）**

```
shock_series：SPY=-0.3, TLT=-0.3, GLD=-0.3（9个资产都有值）

原始 returns：
           SPY     TLT     GLD
Day 1    +0.005  +0.002  -0.001
Day 2    -0.003  +0.004  +0.002
Day 3    +0.008  -0.001  +0.003

+ shock_series（每列对齐后广播到所有行）：

           SPY      TLT      GLD
Day 1    -0.295   -0.298   -0.301
Day 2    -0.303   -0.296   -0.298
Day 3    -0.292   -0.301   -0.297
```

**示例二：Rising Interest Rates（只有部分资产有 shock）**

```
shock_series：TLT=-0.15（只定义了2个资产，其余未定义）

pandas 对齐时：
- 有对应值的列（TLT）：-0.15 广播到所有 N 行 ✅
- 没有对应值的列（SPY、GLD...）：找不到匹配 → 变成 NaN ❌
```

**缺陷与修复**：未指定的资产变成 NaN，会导致后续 mean()、std() 等计算结果异常。
正确写法应在转换时用 0 填充缺失资产：

```python
# 原代码（有 NaN 问题）
shock_series = pd.Series(shock)

# 修复写法：未指定的资产补 0，表示不施加冲击
shock_series = pd.Series(shock).reindex(returns.columns, fill_value=0)
```

---

## 四、每个场景重新计算四项指标

```python
scenario_results[scenario] = {
    'Mean Return': scenario_portfolio_returns.mean(),
    'Volatility':  scenario_portfolio_returns.std(),
    'VaR 95%':    np.percentile(scenario_portfolio_returns, 5),
    'CVaR 95%':   scenario_portfolio_returns[
                      scenario_portfolio_returns <= np.percentile(scenario_portfolio_returns, 5)
                  ].mean()
}
```

对每个场景下的 `shocked_returns` 计算组合日收益率，再算这四项，最终拼成一张对比表。

---

## 五、关于这里的波动率：`std()` vs `std() * √252`

此处计算波动率直接用了 `std()`，没有乘以 `√252`，与前面风险收益分析中的年化波动率不同：

| | 公式 | 单位 |
|--|------|------|
| 场景分析 | `std()` | 日波动率 |
| 风险收益分析 | `std() * √252` | 年化波动率 |

两者本质相同，都是标准差，只是时间单位不同。场景分析做的是各场景之间的**横向对比**，只要所有场景用同一种算法，相对大小就是正确的，不需要年化。

---

## 六、教学简化版的局限性

**Market Crash 场景下，波动率和 Base Case 完全一样：**

```
原始某三天收益：+0.5%, -0.3%, +0.8%
施加 -30% 冲击：-29.5%, -30.3%, -29.2%

std() 衡量数据的"离散程度"（各值偏离均值的幅度）：
原始：各值间距不变 → std = 0.0046
冲击后：整体下移 -30%，但各值间距完全不变 → std 仍 = 0.0046
```

**加一个常数只改变均值，不改变标准差。**

真实的市场崩盘中，恐慌情绪会导致每天涨跌幅急剧放大，波动率会同步飙升。
但这段代码的 shock 是固定常数，只能模拟"收益率整体平移"，无法捕捉极端市场中"波动率本身也在膨胀"的效果。

更真实的压力测试做法：
- 直接使用历史极端时期的真实数据（如2008年9月、2020年3月）
- 对波动率本身也施加冲击（如将 std 乘以某个放大系数）

---

## 七、整体流程总结

```
定义 5 种场景（字典：资产名 → 冲击值）
        ↓
pd.Series(shock).reindex(columns, fill_value=0)
        ↓
shocked_returns = returns + shock_series
（pandas 按列名对齐，shock 值广播到该列所有 N 行）
        ↓
重新计算组合日收益率 → Mean Return / Volatility / VaR / CVaR
        ↓
拼成对比表，可视化各场景下的指标差异
```

**场景分析的价值**：在风险真正发生之前，提前了解组合在各种极端情况下的脆弱点，
从而决定是否需要调整权重或引入对冲工具。
