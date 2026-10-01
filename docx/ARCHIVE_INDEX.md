# RCG-Fusion 历史文档归档索引

本目录存放已经被当前论文主文档取代、但仍有追溯价值的阶段报告、实验协议、观测记录和参考审计。

当前有效的论文主文档位于项目根目录：

- `MAIN_WORK_CN.md`：当前唯一的方法章节主稿；
- `EXPERIMENTS_CN.md`：当前唯一的实验章节与实验状态主稿。

归档文件不应覆盖上述两份主文档中的方法定义、实验协议或结论。引用归档结果时，应先检查其对应的方法版本是否仍然有效。

## 00_project

| 文件 | 性质 | 当前状态 |
|---|---|---|
| `README.md` | 2026-10-01之前的项目入口和运行说明 | 归档；代码命令仍可参考，方法叙述以当前主文档为准 |

## 01_observation_history

| 文件 | 性质 | 当前状态 |
|---|---|---|
| `OBSERVATION_EXPERIMENT_CHECKLIST.md` | RCG观测实验清单与完成状态 | 可用作独立问题观测章节的证据来源 |
| `RESULTS_SYNTHESIS.md` | 早期观测结果汇总 | 历史结果；以更晚的完整复验为准 |
| `CROSS_TASK_GO_NO_GO.md` | 跨任务继续研究判定 | 历史决策记录 |
| `STAGE1_COALITION_REPORT.md` | 早期联盟有效模型报告 | 历史阶段报告 |

## 02_method_history

| 文件 | 性质 | 当前状态 |
|---|---|---|
| `RCG_FUSION_IMPLEMENTATION.md` | 早期RCG实现设计 | 已被当前三创新框架取代 |
| `RCG_FUSION_IMPLEMENTATION_REPORT.md` | 早期RCG实现报告 | 历史实现记录 |
| `BENEFIT_CRC_IMPLEMENTATION_REPORT.md` | 条件收益与CRC硬切换版本 | 失败/历史版本 |
| `POSTERIOR_ANALYTIC_IMPLEMENTATION_REPORT.md` | 解析后验贡献版本 | 部分思想保留在当前创新二 |
| `PROJECTED_FUSION_IMPLEMENTATION_REPORT.md` | 凸包投影版本 | 历史版本 |
| `POSTERIOR_SHRINKAGE_IMPLEMENTATION_REPORT.md` | 后验收缩版本 | 部分思想保留在当前创新三 |
| `RCG_10PCT_REDESIGN.md` | 面向较大性能提升的重设计记录 | 历史设计依据 |
| `LISTWISE_ROUTER_IMPLEMENTATION_REPORT.md` | 软联盟与列表式路由实现诊断 | 当前创新一、二的重要开发依据，但数值属于开发结果 |
| `INTEGRATED_THREE_INNOVATIONS_REPORT.md` | 三创新最近一次联调报告 | 当前主文档的数据来源；仍属于开发结果而非正式论文结论 |

## 03_experiment_history

| 文件 | 性质 | 当前状态 |
|---|---|---|
| `EXPERIMENT_PROTOCOL_V1.md` | 第一版冻结实验协议 | 已过期，旧创新三不再使用 |
| `MAIN_EXPERIMENT_PROGRESS.md` | 旧版E1–E9实验阶段报告 | 已被根目录`EXPERIMENTS_CN.md`取代 |
| `METHOD_PILOT_REPORT.md` | 早期方法先导实验 | 历史开发结果 |

## 04_reference_audits

| 文件 | 性质 | 当前状态 |
|---|---|---|
| `BASELINE_ADAPTATION_AUDIT.md` | 强基线代码与适配审计 | 后续复现时可继续参考 |
| `RELATED_WORK_MATRIX.md` | 相关工作差异矩阵 | 可继续参考，但投稿前需重新核验文献 |
| `FIGURE_CONTRACT.md` | 早期观测图片规范 | 图片技术规范仍可参考，最终图表安排以`EXPERIMENTS_CN.md`为准 |

## 使用规则

1. 新的方法定义只写入根目录`MAIN_WORK_CN.md`。
2. 新的实验协议、状态和论文结果只写入根目录`EXPERIMENTS_CN.md`。
3. 阶段性调试报告如确有必要，应直接放入本归档目录对应分类，不再放回项目根目录。
4. 历史文档中的测试集诊断数字不得自动升级为正式结果。
5. 若主文档引用归档文件，应写明其属于观测证据、开发结果或失败经验。
