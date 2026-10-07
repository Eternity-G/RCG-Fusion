# I2-3 上下文依赖机制实验

目标模态表示保持逐元素不变，只对其余模态施加共享的特征空间高斯扰动。

## 预注册判定

| criterion | datasets_passed | required | pass |
|---|---|---|---|
| fixed-target reliability is invariant (max change <1e-6) | 4 | 4 | True |
| analytic delta-Spearman exceeds reliability by >0.10 | 4 | 3 | True |
| analytic paired-change AUROC >0.65 | 4 | 3 | True |
| strict contribution switches observed | 4 | 3 | True |

## 同一骨干摘要

| dataset | track | method | score_context_mad | score_context_max | contribution_context_mad | delta_spearman | delta_direction_accuracy | delta_auroc | endpoint_auroc | strict_switch_rate | strict_switch_count | strict_switch_endpoint_auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| avmnist | same_backbone | analytic_expected | 0.246873 | 1.962677 | 1.112100 | 0.665328 | 0.860211 | 0.909444 | 0.963495 | 0.007929 | 2569 | 0.665594 |
| avmnist | same_backbone | analytic_probability | 0.188428 | 0.999999 | 1.112100 | -0.355117 | 0.322281 | 0.448291 | 0.994304 | 0.007929 | 2569 | 0.671842 |
| avmnist | same_backbone | anchored_listwise | 0.245354 | 1.978855 | 1.112100 | 0.669389 | 0.869566 | 0.909126 | 0.961671 | 0.007929 | 2569 | 0.667667 |
| avmnist | same_backbone | max_probability | 0.000000 | 0.000000 | 1.112100 | 0.000000 | 0.000000 | 0.500000 | 0.611430 | 0.007929 | 2569 | 0.522520 |
| avmnist | same_backbone | negative_entropy | 0.000000 | 0.000000 | 1.112100 | 0.000000 | 0.000000 | 0.500000 | 0.614350 | 0.007929 | 2569 | 0.527293 |
| cremad | same_backbone | analytic_expected | 0.098741 | 1.530592 | 0.221608 | 0.425064 | 0.683135 | 0.726933 | 0.651570 | 0.056993 | 38173 | 0.683520 |
| cremad | same_backbone | analytic_probability | 0.094247 | 0.995420 | 0.221608 | -0.158647 | 0.393155 | 0.415657 | 0.766332 | 0.056993 | 38173 | 0.790554 |
| cremad | same_backbone | anchored_listwise | 0.097802 | 1.453925 | 0.221608 | 0.323428 | 0.642763 | 0.675380 | 0.667412 | 0.056993 | 38173 | 0.632618 |
| cremad | same_backbone | max_probability | 0.000000 | 0.000000 | 0.221608 | 0.000000 | 0.000000 | 0.500000 | 0.587688 | 0.056993 | 38173 | 0.510187 |
| cremad | same_backbone | negative_entropy | 0.000000 | 0.000000 | 0.221608 | 0.000000 | 0.000000 | 0.500000 | 0.606619 | 0.056993 | 38173 | 0.493371 |
| mosei | same_backbone | analytic_expected | 0.025599 | 0.839945 | 0.043876 | 0.458977 | 0.716898 | 0.781129 | 0.811098 | 0.013557 | 6651 | 0.620995 |
| mosei | same_backbone | analytic_probability | 0.107301 | 0.998164 | 0.043876 | -0.252322 | 0.333019 | 0.363447 | 0.931614 | 0.013557 | 6651 | 0.590069 |
| mosei | same_backbone | anchored_listwise | 0.025193 | 0.763686 | 0.043876 | 0.372141 | 0.673105 | 0.731118 | 0.803352 | 0.013557 | 6651 | 0.609544 |
| mosei | same_backbone | max_probability | 0.000000 | 0.000000 | 0.043876 | 0.000000 | 0.000000 | 0.500000 | 0.769325 | 0.013557 | 6651 | 0.510284 |
| mosei | same_backbone | negative_entropy | 0.000000 | 0.000000 | 0.043876 | 0.000000 | 0.000000 | 0.500000 | 0.769325 | 0.013557 | 6651 | 0.510286 |
| mosi | same_backbone | analytic_expected | 0.021414 | 0.764074 | 0.043538 | 0.324509 | 0.664056 | 0.696992 | 0.727786 | 0.016509 | 1462 | 0.595486 |
| mosi | same_backbone | analytic_probability | 0.106376 | 0.974249 | 0.043538 | -0.265709 | 0.318431 | 0.334940 | 0.884448 | 0.016509 | 1462 | 0.652267 |
| mosi | same_backbone | anchored_listwise | 0.022115 | 0.802039 | 0.043538 | 0.298494 | 0.648879 | 0.684726 | 0.731003 | 0.016509 | 1462 | 0.617097 |
| mosi | same_backbone | max_probability | 0.000000 | 0.000000 | 0.043538 | 0.000000 | 0.000000 | 0.500000 | 0.757502 | 0.016509 | 1462 | 0.508761 |
| mosi | same_backbone | negative_entropy | 0.000000 | 0.000000 | 0.043538 | 0.000000 | 0.000000 | 0.500000 | 0.757503 | 0.016509 | 1462 | 0.508761 |

## 配对比较

| dataset | method | reference | metric | difference | improved_seeds | p | holm_p |
|---|---|---|---|---|---|---|---|
| mosi | analytic_expected | best_unimodal_reliability | delta_spearman | 0.324509 | 5 | 0.000101 | 0.000705 |
| mosi | analytic_probability | best_unimodal_reliability | delta_spearman | -0.265709 | 0 | 0.999643 | 1.000000 |
| mosi | anchored_listwise | best_unimodal_reliability | delta_spearman | 0.298494 | 5 | 0.000194 | 0.001164 |
| mosei | analytic_expected | best_unimodal_reliability | delta_spearman | 0.458977 | 5 | 0.000028 | 0.000222 |
| mosei | analytic_probability | best_unimodal_reliability | delta_spearman | -0.252322 | 0 | 0.999997 | 1.000000 |
| mosei | anchored_listwise | best_unimodal_reliability | delta_spearman | 0.372141 | 5 | 0.001021 | 0.005107 |
| cremad | analytic_expected | best_unimodal_reliability | delta_spearman | 0.425064 | 5 | 0.000001 | 0.000007 |
| cremad | analytic_probability | best_unimodal_reliability | delta_spearman | -0.158647 | 0 | 0.999997 | 1.000000 |
| cremad | anchored_listwise | best_unimodal_reliability | delta_spearman | 0.323428 | 5 | 0.000002 | 0.000020 |
| avmnist | analytic_expected | best_unimodal_reliability | delta_spearman | 0.665328 | 5 | 0.000000 | 0.000006 |
| avmnist | analytic_probability | best_unimodal_reliability | delta_spearman | -0.355117 | 0 | 0.999747 | 1.000000 |
| avmnist | anchored_listwise | best_unimodal_reliability | delta_spearman | 0.669389 | 5 | 0.000000 | 0.000006 |

外部原生分数只在各自模型的贡献事件上作内部诊断，不与同一骨干轨道直接排名。