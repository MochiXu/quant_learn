# 量化交易学习笔记 · 绩效归因（Performance Attribution）

---

## 一、这节在做什么？

**绩效归因回答：组合的总收益里，每个资产各自贡献了多少？**

前面我们计算了组合整体的年化收益率，但不知道这个收益从哪里来。
绩效归因把总收益拆解到每一个资产上，让你看清楚哪个资产是功臣，哪个是拖累。

---

## 二、核心公式

```
每个资产的绝对贡献 = 该资产年化收益率 × 它在组合里的权重
每个资产的相对贡献 = 绝对贡献 ÷ 组合总收益
```

**计算示例（3个资产简化）：**

```
资产      年化收益率    权重      绝对贡献
SPY        +12%     × 0.35  =  +4.20%
TLT        +3%      × 0.25  =  +0.75%
GLD        +8%      × 0.15  =  +1.20%
其余资产合计贡献              =  +1.85%

组合总收益 = 4.20 + 0.75 + 1.20 + 1.85 = 8.00%

资产      绝对贡献    相对贡献（占比）
SPY        4.20%   ÷ 8.00%  =  52.5%  ← 贡献了一半以上
GLD        1.20%   ÷ 8.00%  =  15.0%
TLT        0.75%   ÷ 8.00%  =   9.4%
```

---

## 三、代码解析

```python
def performance_attribution(weights, returns):
    # 组合年化收益率（标量）
    portfolio_return = np.sum(returns.mean() * weights) * 252

    # 每个资产对组合的年化绝对贡献（向量，9个值）
    asset_contribution = returns.mean() * weights * 252

    # 各资产贡献占组合总收益的比例（向量，9个值）
    percent_contribution = asset_contribution / portfolio_return

    attribution_data = pd.DataFrame({
        'Weight':               weights,
        'Return':               returns.mean() * 252,
        'Contribution':         asset_contribution,
        'Percent Contribution': percent_contribution
    })
    return attribution_data.sort_values('Percent Contribution', ascending=False)
```

---

## 四、`returns.mean() * weights` 的计算原理

### NumPy 的 `(9,)` 是什么？

NumPy 严格区分三种形状：

```
(9,)    → 1D 数组，没有行列方向，就是"9个数的扁平列表"
(9, 1)  → 2D 列向量，明确是 9行1列
(1, 9)  → 2D 行向量，明确是 1行9列
```

`returns` 是 `(N, 9)` 的 DataFrame，调用 `.mean()` 后沿 axis=0（按列）压缩：

```
(N, 9)  →  .mean()  →  (9,)

每列 N 个日收益率 → 压缩成 1 个均值
结果是形状 (9,) 的 Series，9个资产各对应一个均值
不是 (9, 1)，不是 (1, 9)，是没有行列方向的 1D 数组
```

### `*` 是逐元素相乘，不是矩阵乘法

```
returns.mean()  形状 (9,)
weights         形状 (9,)

* 号 = Element-wise（逐元素相乘），对应位置相乘
结果仍然是 (9,)，保留9个独立的值
```

```
returns.mean()   *   weights   =   asset_contribution
  SPY  0.0004   ×   0.35      =   0.000140
  IWM  0.0003   ×   0.20      =   0.000060
  GLD  0.0002   ×   0.15      =   0.000030
  ...                              ...（共9项，各自独立）
```

### 为什么两行代码结果一个是标量，一个是向量？

代码里有两行长得相似，关键差异是有没有 `np.sum()`：

```python
# portfolio_return：有 np.sum()，把 9 个值加总 → 标量
portfolio_return = np.sum(returns.mean() * weights) * 252

# asset_contribution：没有 np.sum()，保留 9 个值 → 形状 (9,)
asset_contribution = returns.mean() * weights * 252
```

```
逐元素相乘后的 (9,) 向量：
[0.000140, 0.000060, 0.000030, ...]

加上 np.sum() → 全部加起来 = 0.00032（标量）= 组合日均收益
不加 np.sum() → 保留 [0.000140, 0.000060, ...]（向量）× 252 = 各资产年化贡献
```

绩效归因恰恰需要保留每个资产独立的贡献值，所以 `asset_contribution` 不能加 `np.sum()`。

### `*` vs `np.dot()` 的本质区别

```python
a = np.array([1, 2, 3])   # 形状 (3,)
b = np.array([4, 5, 6])   # 形状 (3,)

a * b        # 逐元素相乘 → [4, 10, 18]，形状 (3,)，保留3个值
np.dot(a, b) # 点积       → 4+10+18 = 32，标量，结果只有1个值
```

| 运算 | 符号 | 结果形状 | 用途 |
|------|------|----------|------|
| 逐元素相乘 | `*` | `(9,)` | 保留每个资产的独立贡献 |
| 点积 | `np.dot()` | 标量 | 求组合整体的加权总和 |

---

## 五、负贡献的含义

`Percent Contribution` 可能出现负数：

```
EEM 年化收益率 = -2%，权重 = 0.05
EEM 绝对贡献 = -2% × 0.05 = -0.1%
Percent Contribution = -0.1% ÷ 8.0% = -1.25%

→ 这个资产不仅没有贡献收益，还拖累了整体组合
```

---

## 六、这个分析的实际用途

```
Percent Contribution 排名靠前 → 组合收益高度依赖这个资产，集中度风险高
Percent Contribution 为负     → 这个资产是当前配置下的拖累项，考虑减仓
权重大但贡献小                → 该资产收益率低，配置性价比差
```

**例：若 SPY 贡献超过 50%，说明这个"多资产组合"实际上高度依赖美股，
分散化效果有限，可考虑降低 SPY 权重或增加低相关资产。**

绩效归因的价值：不只是看组合总收益，还要知道收益**从哪里来**，
才能做出更有依据的调整决策。
