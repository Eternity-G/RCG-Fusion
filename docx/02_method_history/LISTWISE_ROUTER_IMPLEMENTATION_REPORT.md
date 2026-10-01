# 软联盟目标与列表式路由器实现报告

## 已实现内容

### 创新一：OOF软联盟监督

`rcg/listwise_router.py` 新增 `build_soft_coalition_targets`，从未见过对应样本的教师输出
生成：

- 温度化软 oracle 分布；
- 全联盟成对偏序概率；
- 相对完整融合的稳定改善概率和80%一致标签；
- 基于软分布熵的教师一致性权重；
- 平均损失与平均 oracle 联盟。

目标保存在：

```text
runs/<run>/fold_<k>/soft_coalition_targets.npz
```

其中包含 `sample_id`、`group_id`、OOF fold、teacher seeds 和所有监督数组。

### 创新二：解析先验加列表式残差

实现 `AnalyticResidualListwiseRouter`：

\[
s_S=u_S^{analytic}+r_\theta(S,x).
\]

残差头零初始化，训练目标由软列表 KL、成对偏序、稳定改善分类、标签后验 proper loss
和残差正则组成。联盟查询不使用位置编码，任意调整联盟查询顺序时输出同步置换。

新增 selection 残差步长：

```text
eta ∈ {0, 0.1, 0.25, 0.5, 0.75, 1}
```

只按 selection 联盟 regret 选择。`eta=0` 是纯解析回退，不使用测试标签。

命令：

```powershell
.\.venv\Scripts\python.exe -m rcg.listwise_router_pipeline `
  --dataset mosi --data data `
  --base-run runs\rcg-fusion-mosi-v4 `
  --output runs\rcg-listwise-mosi-v1
```

可使用冻结后验作为解析先验：

```powershell
  --posterior-run runs\rcg-posterior-analytic-mosi-v2
```

## 软件验收

当前测试为 **73 passed**。新增检查包括：

- 软联盟分布严格归一化；
- 成对偏序方向正确；
- 残差零初始化时严格等于解析先验；
- 列表式总损失有限；
- 联盟查询置换等变；
- 完美排序的 Top-1/Top-2/NDCG/成对准确率等于1；
- selection 可以将残差步长回退为0。

## MOSI五种子结果

### 联合学习标签后验

| 指标 | 解析先验 | 列表式残差 | 差值 |
|---|---:|---:|---:|
| Top-1 | 51.59% | 43.87% | -7.71pp |
| Top-2 | 53.20% | 55.12% | +1.92pp |
| 成对准确率 | 71.83% | 71.57% | -0.26pp |
| 选择Accuracy | — | 75.95% | Base 77.87% |

### 冻结五模型后验、多教师上下文

| 指标 | 解析先验 | 列表式残差 | 差值 |
|---|---:|---:|---:|
| Top-1 | 33.45% | 28.32% | -5.12pp |
| Top-2 | 36.28% | 44.33% | +8.05pp |
| 成对准确率 | 72.55% | 70.88% | -1.67pp |
| 选择Accuracy | — | 78.08% | Base 77.87% |

因此当前版本只改善 Top-2 候选覆盖，未改善 Top-1 和成对排序，没有通过预注册门槛，
不能进入创新三正式训练。

## 失败原因核查

MOSI训练集中，五个 OOF 教师的 oracle 联盟一致性为：

| 统计 | 数值 |
|---|---:|
| 平均最大投票比例 | 65.05% |
| 至少4/5教师一致 | 36.80% |
| 五教师完全一致 | 11.54% |
| 任意两教师oracle完全相同 | 44.58% |

逐样本硬最优联盟对训练随机性非常敏感。软目标缓解了问题并提高 Top-2，但不足以支撑
稳定 Top-1 路由。继续扩大网络或增加训练轮次更可能拟合教师噪声。

## 研究判断

创新一的软件与监督接口已经完成，但实验揭示“精确恢复逐样本最佳联盟”不是稳定的主
学习目标。下一版应把列表式排序降为辅助任务，以“是否稳定优于完整融合”的多标签集合
预测作为主任务；最终融合对 Top-2/Top-3 稳定改善联盟进行风险受控混合，而不依赖单个
Top-1 oracle。

