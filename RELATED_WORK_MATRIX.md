# 条件贡献信息融合：相关工作与差异边界（2026-09-27）

## 审计结论

当前可辩护的研究空白不是“动态选择模态”“建模跨模态关系”或“首次使用 Shapley 分析模态”。这些能力分别已被 QMF/PDF、AEMRL/I²MoE/QA-MoE 和 SHAPE 覆盖。

本项目应把主张限定为：

> 用训练标签定义样本级、带符号的反事实联盟损失差，通过交叉拟合学习测试时无标签的联盟风险，并显式最小化相对于最佳可用联盟的融合后悔值。

截至本次审计，未发现下列三项同时成立的正式工作：

1. 以样本级联盟反事实损失作为动态融合监督；
2. 预测带符号条件贡献或全部联盟任务损失，而非单模态质量/一般交互权重；
3. 以完整融合相对于最佳联盟的后悔值作为核心训练与评价对象。

这是文献检索后的候选差异点，不是对“首次”的最终保证；投稿前仍需做引文前向追踪。

## 逐项矩阵

| 工作 | 估计/优化对象 | 关系随上下文变化 | 允许负贡献 | 拒绝完整融合 | 测试标签 | 删除伪象控制 | 样本级贡献 | 代码审计 | 与本项目的实质边界 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| [TMC, ICLR 2021](https://github.com/Han-Zongbo/TMC) | 每视图 Dirichlet evidence 与 DS 组合 | 部分 | 否 | 否 | 否 | 不适用 | 否 | 官方代码可用；本项目已完成统一特征适配 | TMC 的可靠性仍是视图证据强度，不直接监督条件联盟损失 |
| [SHAPE, IJCAI 2022](https://www.ijcai.org/proceedings/2022/425) | Shapley 模态贡献与合作分数 | 是 | 数学上可以 | 否 | 离线评价使用标签 | 使用置换/基线构造 | 主要用于评价 | 论文与公式可用；未找到维护中的官方实现 | 本项目不能声称首次使用 Shapley；区别是带符号样本损失、联盟有效训练、贡献预测和后悔决策 |
| [QMF, ICML 2023](https://proceedings.mlr.press/v202/zhang23ar.html) | 单模态 uncertainty/energy 与动态权重 | 有限 | 否 | 否 | 否 | 不适用 | 否 | [官方代码](https://github.com/QingyangZhang/QMF)可用，覆盖图文与 RGB-D | QMF 假设 uncertainty 与单模态 loss 相关；本项目检验并学习相对联盟损失 |
| [PDF, ICML 2024](https://proceedings.mlr.press/v235/cao24c.html) | Mono-/Holo-Confidence 与 Co-Belief | 是 | 未显式建模带符号贡献 | 否 | 否 | 不适用 | 置信度是样本级 | [官方代码](https://github.com/Yinan-Xia/PDF)可用，环境较旧 | 必须直接比较 Co-Belief 对负贡献和最佳联盟的识别能力 |
| [I²MoE, ICML 2025](https://proceedings.mlr.press/v267/xin25c.html) | uniqueness/redundancy/synergy interaction experts 与重加权 | 是 | 交互可区分，但不是损失贡献符号 | 可以稀疏路由专家 | 否 | 未以联盟有效删除为核心 | 是 | [官方代码](https://github.com/Raina-Xin/I2MoE)可用，含 MOSI/MM-IMDb/Enrico | 不能声称首次建模冗余和协同；需要证明其 interaction weight 不等于反事实条件贡献 |
| [AEMRL, IJCAI 2026](https://www.ijcai.org/proceedings/2026/128) | intrinsic、relational、aggregated uncertainty | 是 | 未直接定义任务损失负贡献 | 冲突抑制，但不是联盟 oracle | 否 | 未见联盟训练说明 | 是 | 官方论文可用；本次检索未定位官方代码 | 本项目不得复用“三级不确定性”作为新颖叙事，应聚焦 signed utility/regret |
| [QA-MoE, ACL 2026](https://aclanthology.org/2026.acl-long.1461/) | 自监督 aleatoric quality 与 MoE routing | 质量随输入变化 | 否 | 可抑制低质量模态 | 否 | 通过连续退化训练 | 是 | 正式论文可用；本次检索未定位官方代码 | 这是 MOSI/MOSEI 上最接近的质量路由基线；核心差异必须是高质量但负贡献与反事实监督 |
| [QA-MoE, IJCAI 2026](https://www.ijcai.org/proceedings/2026/766) | Evidential Quality Scorer 与稳定子集选择 | 是 | 否 | 是 | 否 | 缺失/噪声训练 | 是 | 正式论文可用；本次检索未定位官方代码 | 已覆盖“质量评分+子集选择”，因此单纯 selector 不能成为本项目创新 |

## 必须采用的基线层级

1. **静态与协议对照**：等权概率、完整输入 concat、coalition-dropout concat、oracle 最佳联盟。
2. **可靠性融合**：TMC、QMF、PDF。
3. **关系与路由**：I²MoE、AEMRL、ACL 2026 QA-MoE、IJCAI 2026 QA-MoE。
4. **诊断参照**：SHAPE/精确 Shapley，但不把它当作预测方法。

若官方代码不可用，统一特征适配只能称为 adaptation，不得用其数值声称复现原论文结果。所有方法都要额外报告负贡献识别、最佳联盟恢复和融合后悔值，不能只比较 Accuracy/F1。

## 当前新颖性风险

- **高风险**：把“跨模态关系不确定性”称为创新，AEMRL 已明确提出。
- **高风险**：把“质量感知子集选择”称为创新，两篇不同的 QA-MoE 已覆盖。
- **高风险**：把“交互的唯一性/冗余/协同”称为创新，I²MoE 已覆盖。
- **中风险**：把样本级 Shapley 本身称为创新，SHAPE 已奠定模态 Shapley 评价。
- **可保留方向**：训练期反事实联盟损失监督、无标签联盟风险预测、带符号贡献识别与低后悔决策的统一闭环。

