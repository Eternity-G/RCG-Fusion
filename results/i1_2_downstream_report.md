# I1-2 软监督下游传递

- 至少两个数据集改善下游NLL：通过。
- 所有版本使用同一骨干、后验、mixer、收缩控制器和selection协议。

| dataset | variant | nll_gain_soft | accuracy_gain | regret_reduction_soft | correction | negative_flip | clipped_harm | strength | nll_gain_hard | regret_reduction_hard | nll_transfer_advantage | regret_transfer_advantage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| avmnist | oof_soft_full | 0.00397 | 0.00111 | 0.17749 | 0.00167 | 0.00056 | 0.00307 | 0.95000 | 0.00397 | 0.17749 | 0.00000 | 0.00000 |
| cremad | oof_soft_full | 0.02599 | 0.01053 | 0.07462 | 0.01854 | 0.00801 | 0.01608 | 0.49905 | 0.02599 | 0.07462 | 0.00000 | 0.00000 |
| mosei | oof_soft_full | 0.00387 | 0.00270 | 0.02398 | 0.00759 | 0.00490 | 0.00816 | 1.00000 | 0.00365 | 0.02261 | 0.00023 | 0.00138 |
| mosi | oof_soft_full | 0.00645 | 0.00701 | 0.03918 | 0.01433 | 0.00732 | 0.00904 | 0.85000 | 0.00608 | 0.03701 | 0.00038 | 0.00217 |

配对比较及Holm校正见 `i1_2_downstream_comparisons.csv`。