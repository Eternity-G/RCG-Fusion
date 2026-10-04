# I1-1 OOF监督可信性实验

- 状态：正式复核（复用冻结的四数据集五教师、五路由种子产物）。
- OOF乐观偏差检查：通过。
- 软目标方差降低：通过。
- 至少两个非饱和数据集保留决策信息：通过。
- Holm校正后显著性与效应量见 `i1_1_supervision_comparisons.csv`。

## 数据集级审计

| dataset | mean_in_sample_loss | mean_oof_loss | oof_minus_in_sample_loss | oof_in_sample_loss_spearman | hard_target_variance | soft_target_variance | soft_to_hard_variance_ratio | mean_max_vote_fraction | at_least_four_of_five_agreement | five_of_five_agreement | mean_pairwise_teacher_oracle_agreement | normalized_soft_oracle_entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mosi | 0.48216 | 0.57924 | 0.09707 | 0.75377 | 0.44335 | 0.03327 | 0.07503 | 0.65053 | 0.36799 | 0.11535 | 0.44582 | 0.76969 |
| mosei | 0.44130 | 0.49369 | 0.05239 | 0.89111 | 0.39439 | 0.02254 | 0.05716 | 0.69071 | 0.45346 | 0.18473 | 0.50702 | 0.75439 |
| cremad | 0.88611 | 1.29467 | 0.40857 | 0.63082 | 0.22491 | 0.12214 | 0.54422 | 0.83045 | 0.72320 | 0.46073 | 0.71886 | 0.47064 |
| avmnist | 0.21845 | 0.26140 | 0.04296 | 0.86427 | 0.05089 | 0.02122 | 0.41706 | 0.96333 | 0.95131 | 0.86625 | 0.93639 | 0.81073 |

## 固定完整软监督相对OOF多教师硬目标

| dataset | coverage_gain | oracle_nll_gain |
|---|---|---|
| avmnist | 0.00011 | 0.00044 |
| cremad | -0.00150 | 0.00037 |
| mosei | 0.11128 | 0.00450 |
| mosi | -0.08201 | 0.00049 |

## 判读

软监督在全部数据集都显著降低教师目标方差，但候选学习收益具有任务依赖性。因此I1-1支持其作为泄漏受控、保留教师不确定性的监督机制；它能否传递到最终决策必须由I1-2单独验证。