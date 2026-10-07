# I2-2 列表式残差路由与下游传递

候选排序和最终决策使用相同骨干、联盟概率、后验、mixer及收缩控制器。
列表路由只有在候选覆盖能够传递到决策收益时才计作系统性能证据。

## 预注册判定

| criterion | datasets_passed | required | pass |
|---|---|---|---|
| MOSI/MOSEI candidate recovery gain > 2pp with Holm p<0.05 | 2 | 2 | True |
| anchored routing improves downstream NLL gain | 2 | 2 | True |
| saturated tasks lose no more than 0.0005 NLL gain | 2 | 2 | True |
| anchoring does not reduce mean downstream gain by >0.0005 | 4 | 4 | True |

## 主要配对比较

| dataset | second | first | metric | difference_second_minus_first | improved_seeds | n_seeds | paired_t_one_sided_p | family | holm_p |
|---|---|---|---|---|---|---|---|---|---|
| mosi | listwise_pairwise_anchored | analytic_only | route_candidate_oracle_recovery | 0.137500 | 5 | 5 | 0.020257 | non_saturated_candidate_recovery | 0.040514 |
| mosei | listwise_pairwise_anchored | analytic_only | route_candidate_oracle_recovery | 0.182664 | 4 | 5 | 0.021838 | non_saturated_candidate_recovery | 0.040514 |
| mosi | listwise_pairwise_anchored | analytic_only | adaptive_nll_gain | 0.000024 | 2 | 5 | 0.451441 | downstream_nll_transfer | 1.000000 |
| mosei | listwise_pairwise_anchored | analytic_only | adaptive_nll_gain | 0.000173 | 3 | 5 | 0.088272 | downstream_nll_transfer | 0.353087 |
| cremad | listwise_pairwise_anchored | analytic_only | adaptive_nll_gain | 0.000000 | 0 | 5 | 1.000000 | downstream_nll_transfer | 1.000000 |
| avmnist | listwise_pairwise_anchored | analytic_only | adaptive_nll_gain | 0.000000 | 0 | 5 | 1.000000 | downstream_nll_transfer | 1.000000 |
| mosi | listwise_pairwise_anchored | listwise_pairwise_unanchored | route_action_oracle_nll | -0.002593 | 2 | 5 | 0.097026 | anchoring_action_oracle | 0.388104 |
| mosei | listwise_pairwise_anchored | listwise_pairwise_unanchored | route_action_oracle_nll | 0.000393 | 1 | 5 | 0.851275 | anchoring_action_oracle | 1.000000 |
| cremad | listwise_pairwise_anchored | listwise_pairwise_unanchored | route_action_oracle_nll | 0.000250 | 3 | 5 | 0.894326 | anchoring_action_oracle | 1.000000 |
| avmnist | listwise_pairwise_anchored | listwise_pairwise_unanchored | route_action_oracle_nll | 0.000000 | 0 | 5 | 1.000000 | anchoring_action_oracle | 1.000000 |

## 锚定列表式路由摘要

| dataset | variant | ndcg | pairwise_accuracy | candidate_recovery | candidate_oracle_nll | candidate_headroom_fraction | adaptive_nll_gain | adaptive_accuracy_gain | correction_rate | negative_flip_rate | clipped_harm | regret_reduction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| avmnist | listwise_pairwise_anchored | 0.995185 | 0.986744 | 0.993556 | 0.030054 | 0.502565 | 0.003974 | 0.001111 | 0.001667 | 0.000556 | 0.003072 | 0.177492 |
| cremad | listwise_pairwise_anchored | 0.848892 | 0.725074 | 0.808949 | 0.870039 | 0.530095 | 0.025990 | 0.010535 | 0.018543 | 0.008009 | 0.016079 | 0.074621 |
| mosei | listwise_pairwise_anchored | 0.893288 | 0.791515 | 0.671271 | 0.334234 | 0.239584 | 0.003844 | 0.003027 | 0.007870 | 0.004843 | 0.008250 | 0.023829 |
| mosi | listwise_pairwise_anchored | 0.847288 | 0.717938 | 0.535366 | 0.422407 | 0.312060 | 0.006349 | 0.006098 | 0.013415 | 0.007317 | 0.009088 | 0.038586 |

## 解析排序摘要

| dataset | variant | ndcg | pairwise_accuracy | candidate_recovery | candidate_oracle_nll | candidate_headroom_fraction | adaptive_nll_gain | adaptive_accuracy_gain | correction_rate | negative_flip_rate | clipped_harm | regret_reduction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| avmnist | analytic_only | 0.995244 | 0.987831 | 0.993722 | 0.029999 | 0.502566 | 0.003974 | 0.001111 | 0.001667 | 0.000556 | 0.003072 | 0.177492 |
| cremad | analytic_only | 0.839370 | 0.715853 | 0.802204 | 0.874474 | 0.521310 | 0.025990 | 0.010535 | 0.018543 | 0.008009 | 0.016079 | 0.074621 |
| mosei | analytic_only | 0.885659 | 0.795961 | 0.488608 | 0.338903 | 0.207460 | 0.003671 | 0.002422 | 0.007430 | 0.005008 | 0.008171 | 0.022770 |
| mosi | analytic_only | 0.843955 | 0.725494 | 0.397866 | 0.426693 | 0.292814 | 0.006325 | 0.006098 | 0.013110 | 0.007012 | 0.008664 | 0.038419 |

MOSI/MOSEI属于候选非饱和任务；CREMA-D和AV-MNIST的解析候选覆盖已较高。
结构性Top-1锚定本身不是经验性能结果，正文只将其作为候选保护机制。