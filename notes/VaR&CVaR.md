# 量化交易学习笔记 · VaR & CVaR（高级风险指标）

---

## 一、核心问题

VaR 和 CVaR 共同回答一个问题：**最坏的情况下，我一天会亏多少？**

---

## 二、VaR（Value at Risk，风险价值）

**定义：在给定置信水平下，某一天的最大预期亏损。**

以 95% 置信水平为例：

```
95% VaR = -1.2%

含义：在历史数据中，有 95% 的交易日，单日亏损不会超过 1.2%。
     只有 5% 的交易日，亏损会比 1.2% 更严重。
```

**直觉理解**：把所有历史日收益率从小到大排列，VaR 就是站在最差那端 5% 处的分界线。

```
频率
 ↑
 │        ╭──╮
 │       ╭╯  ╰╮
 │      ╭╯    ╰──╮
 │  ╭───╯        ╰───╮
 └──┼────────────────┼──→ 日收益率
  最差             最好
  ↑
  VaR（第 5% 分位数）
  左侧 5% 的天数比这更糟糕
```

---

## 三、CVaR（Conditional Value at Risk，条件风险价值）

VaR 只给出了亏损门槛，但没说越过这条线之后**平均亏多少**。CVaR 补充了这个信息：

**定义：在亏损超过 VaR 的那些交易日里，平均亏损是多少。**

```
95% CVaR = -1.8%

含义：在最差的 5% 交易日中，平均每天亏损 1.8%。
```

```
频率
 ↑
 │        ╭──╮
 │       ╭╯  ╰╮
 │      ╭╯    ╰──╮
 │  ╭───╯        ╰───╮
 └──┼──┼─────────────┼──→ 日收益率
    ↑  ↑
  CVaR VaR
  （尾部平均值）（分界线）
```

**CVaR 比 VaR 更保守**，衡量的是极端亏损时的平均情况，对尾部风险的刻画更完整。

---

## 四、代码解析

### 4.1 计算组合每日收益率

```python
portfolio_returns = np.sum(returns * weights, axis=1)
```

`returns` 是形状为 `(N, 9)` 的矩阵，N 为交易日数，9 为资产数。
`weights` 是形状为 `(9,)` 的向量。

**实现方式一：NumPy 广播（代码中的写法）**

```
第一步：returns * weights
  returns 形状 (N, 9)，weights 形状 (9,)
  NumPy 广播：weights 自动扩展，对 returns 的每一行逐元素相乘
  结果仍为 (N, 9)

  Day 1: [r1_SPY, r1_IWM, ..., r1_GLD] × [w_SPY, w_IWM, ..., w_GLD]
       = [r1_SPY×w_SPY, r1_IWM×w_IWM, ..., r1_GLD×w_GLD]
  Day 2: [r2_SPY, r2_IWM, ..., r2_GLD] × [w_SPY, w_IWM, ..., w_GLD]
       = [r2_SPY×w_SPY, r2_IWM×w_IWM, ..., r2_GLD×w_GLD]
  ...

第二步：np.sum(..., axis=1)
  axis=1 沿列方向（横向）求和，将 9 列压缩成 1 个数
  (N, 9) → (N,)

  Day 1: r1_SPY×w_SPY + r1_IWM×w_IWM + ... + r1_GLD×w_GLD = 第1天组合收益率
  Day 2: r2_SPY×w_SPY + r2_IWM×w_IWM + ... + r2_GLD×w_GLD = 第2天组合收益率
```

**实现方式二：矩阵乘法（np.dot），结果完全等价**

```python
portfolio_returns = np.dot(returns, weights)
# (N×9) · (9×1) = (N×1)，即每一行与 weights 做点积
```

```
矩阵乘法视角：
  (N × 9) · (9 × 1) = (N × 1)
  每行9个资产收益率 × 对应权重后求和 = 那天的组合收益率
```

两种写法数学结果完全相同，`portfolio_returns` 是长度为 N 的向量，每个元素代表在给定 `weights` 下那一天整个组合的日收益率。

---

### 4.2 计算 VaR

```python
var = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
```

`np.percentile(data, 5)` 找到数据的第 5 百分位数：

```
confidence_level = 0.95
100 * (1 - 0.95) = 5

把 N 天收益率从小到大排列，取排在第 5% 位置的值即为 VaR。

假设共 1000 天数据：
排第 1 名（最差）：-3.1%
排第 2 名：       -2.8%
...
排第 50 名：      -1.2%  ← 第 5 百分位数 = 95% VaR
...
排第 1000 名（最好）：+2.5%
```

---

### 4.3 计算 CVaR

```python
cvar = portfolio_returns[portfolio_returns <= var].mean()
```

```
portfolio_returns <= var        筛选出所有比 VaR 更差的交易日（最差的 5%）
[...].mean()                    计算这些极端亏损日的平均收益率

即：CVaR = 尾部最差 5% 交易日的平均亏损
```

---

### 4.4 可视化

```python
sns.histplot(optimal_portfolio_returns, kde=True)
plt.axvline(var_95,  color='r', linestyle='dashed', label='95% VaR')
plt.axvline(cvar_95, color='g', linestyle='dashed', label='95% CVaR')
```

- `sns.histplot`：绘制日收益率的频率分布直方图
- `kde=True`：在直方图上叠加平滑的概率密度曲线
- `axvline`：在图上画垂直线，红线 = VaR，绿线 = CVaR

最终图示：

```
频率
 ↑
 │         ╭──╮
 │        ╭╯  ╰╮
 │       ╭╯    ╰──╮
 │   ╭───╯        ╰────╮
 └───┼──┼──────────────┼──→ 日收益率
     ↑  ↑           0%
   绿线 红线
  CVaR  VaR
```

---

## 五、VaR vs CVaR 对比总结

| 指标 | 回答的问题 | 局限性 |
|------|-----------|--------|
| **VaR 95%** | 95% 的日子里，最多亏多少？ | 不描述超过门槛后的亏损程度 |
| **CVaR 95%** | 最差的 5% 日子里，平均亏多少？ | 依赖历史数据，黑天鹅事件可能更极端 |

> **CVaR 是风险管理中更受重视的指标**。真正的危险不是知道门槛在哪，
> 而是越过门槛后平均会发生什么。现代机构风控普遍使用 CVaR 作为核心约束指标。
