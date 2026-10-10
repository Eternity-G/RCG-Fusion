# RCG-Fusion 补充实验完成审计

审计日期：2026-10-10  
最终方法版本：`rcg-fusion-a8-v2`

## 1. 执行范围

本审计覆盖 `P0`、`I1-1`、`I1-2`、`I2-1`、`I2-2`、`I2-3`、`I3-1`、`I3-2`、`I3-3`、`S1`、`S2`、`S3`、`S4`、`S5` 和 `S6`。四个数据集均为现有固定表示协议：CMU-MOSI、CMU-MOSEI、CREMA-D 和 AV-MNIST。

## 2. 逐项证据

| 项目 | 状态 | 决定性证据 | 主要产物 |
|---|---|---|---|
| P0 最终系统统一 | 完成 | 四数据集预测均登记 `method_version`、SHA-256 和统一推理路径 | `p0_final_system_registry.csv/json` |
| I1-1 OOF监督可信性 | 完成 | 四数据集六版本；软目标方差降低，OOF隔离和教师分歧已审计 | `i1_1_supervision_*.csv`、`i1_1_supervision_report.md` |
| I1-2 下游传递 | 完成 | 固定后续结构，只替换监督版本；MOSI/MOSEI小幅正传递，CREMA-D/AV-MNIST安全持平 | `i1_2_downstream_*.csv`、`i1_2_downstream_report.md` |
| I2-1 解析贡献识别 | 完成 | AUROC增益、5/5种子、Top-20%富集和十分位顺序四项预注册判据全部通过 | `i2_1_decision.csv`、`i2_1_report.md` |
| I2-2 列表式残差路由 | 完成 | MOSI/MOSEI候选恢复显著提高，饱和任务无明显破坏 | `i2_2_decision.csv`、`i2_2_report.md` |
| I2-3 上下文依赖 | 完成 | 可靠性严格不变；解析贡献在四数据集追踪真实贡献变化 | `i2_3_decision.csv`、`i2_3_report.md` |
| I3-1 候选融合 | 完成 | 完整软mixer在4/4数据集优于最佳硬路由；纠错率均高于负向翻转 | `i3_1_decision.csv`、`i3_1_report.md` |
| I3-2 匹配损害预算 | 完成 | 贡献概率控制4/4进入Pareto前沿，3/4在2%预算胜过简单控制 | `i3_2_decision.csv`、`i3_2_report.md` |
| I3-3 稳定聚合与回退 | 完成 | 3/4优于A7等权、4/4优于完整集成，非负约束降低尾部损害 | `i3_3_decision.csv`、`i3_3_report.md` |
| S1 干净主结果 | 完成 | 全外部基线、四数据集五组独立五成员复验、训练重复CI与配对bootstrap | `s1_clean_*`、`s1_replicates_*`、`s1_replicate_decision.csv` |
| S2 连续退化 | 完成 | clean-only与共享增强两协议，四种五成员强基线，A7/A8分布匹配规则 | `s2_condition_metrics.csv`、`s2_robustness_summary.csv` |
| S3 模态缺失 | 完成 | 全部合法联盟、精确单联盟回退和强基线缺失适配 | `s3_missing_summary.csv`、`s3_missing_bootstrap.csv` |
| S4 骨干迁移 | 完成 | 五骨干、MOSI与CREMA-D；样本级完整RCG在10/10组合降低NLL | `s4_transfer_metrics.csv`、`s4_transfer_bootstrap.csv` |
| S5 三链消融 | 完成 | 监督、路由、决策三条链分别归因，A8统一为P0 v2 | `s5_three_chain_ablation.csv` |
| S6 效率 | 完成 | 单成员、蒸馏、五成员、五类外部骨干以及每0.01 NLL的增量延迟 | `s6_*.csv` |

## 3. S1独立重复结论

| 数据集 | Accuracy增益均值 | Accuracy训练组95% CI | NLL改善均值 | NLL训练组95% CI | NLL胜场 |
|---|---:|---:|---:|---:|---:|
| MOSI | +0.46 pp | [-0.36, +1.27] pp | +0.00957 | [-0.00057, 0.01971] | 4/5 |
| MOSEI | +0.02 pp | [-0.12, +0.16] pp | +0.00042 | [-0.00004, 0.00087] | 5/5 |
| CREMA-D | +0.74 pp | [+0.25, +1.22] pp | +0.01664 | [0.00959, 0.02370] | 5/5 |
| AV-MNIST | +0.07 pp | [-0.06, +0.19] pp | +0.00411 | [0.00203, 0.00620] | 5/5 |

CREMA-D同时获得分类决策和概率质量的跨训练重复支持；AV-MNIST获得概率质量支持。MOSI的效应幅度较大但训练组区间略跨零；MOSEI方向一致但效应很小。

## 4. 未通过的论文级强判据

实验执行完成不等于所有顶会级性能假设均通过。相对每个数据集最强计算匹配外部基线：

- RCG-Fusion在3/4数据集取得最低数值NLL，但单系统配对CI没有在3个数据集排除零；
- Accuracy只在MOSI超过最强Accuracy基线；
- CREMA-D统一骨干的Accuracy低于TMC，但S4中 `TMC + RCG` 能在保留强骨干的条件下降低NLL；
- MOSEI最终增益过小，不能作为主要性能证据；
- 五成员完整系统的推理开销高，蒸馏学生尚未完全恢复教师性能。

因此可直接支撑的核心主张是：三个模块分别在监督可信性、条件贡献识别、候选覆盖、匹配损害预算和跨骨干NLL校正上提供证据；不能声称完整系统在所有数据集上显著提高Accuracy，也不能将小幅MOSEI改善包装为实质提升。

## 5. 完整性检查

- 自动化测试：87项通过；
- 实验章节引用PNG：35张全部存在、300 dpi、白底且alpha全不透明；
- 数据目录由 `.gitignore` 排除；
- 训练检查点和大体积逐样本运行记录保留在本地 `runs/`，不上传GitHub；
- 可复算统计CSV、Markdown报告和论文PNG纳入版本控制。

