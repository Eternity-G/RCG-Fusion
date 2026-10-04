# I2-1 解析贡献识别主实验

同一骨干比较与各方法原生分数诊断保持为两个独立轨道。

## 预注册判定

| criterion | datasets_passed | required | pass |
|---|---|---|---|
| AUROC gain > 0.05 | 4 | 3 | True |
| 5/5 seeds over direct regression | 4 | 3 | True |
| Top-20 precision above prevalence after Holm | 4 | 3 | True |
| positive decile safe-rate ordering | 4 | 4 | True |

## 主要配对比较

| dataset | second | first | metric | difference | improved_seeds | n_seeds | paired_t_one_sided_p | holm_p |
|---|---|---|---|---|---|---|---|---|
| mosi | learned_posterior_analytic | direct_risk_regression | auroc | 0.10586 | 5 | 5 | 0.00000 | 0.00002 |
| mosi | top20_precision | event_prevalence | precision | 0.33527 | 5 | 5 | 0.00000 | 0.00002 |
| mosei | learned_posterior_analytic | direct_risk_regression | auroc | 0.22975 | 5 | 5 | 0.00007 | 0.00022 |
| mosei | top20_precision | event_prevalence | precision | 0.39972 | 5 | 5 | 0.00000 | 0.00000 |
| cremad | learned_posterior_analytic | direct_risk_regression | auroc | 0.16731 | 5 | 5 | 0.00000 | 0.00002 |
| cremad | top20_precision | event_prevalence | precision | 0.29262 | 5 | 5 | 0.00000 | 0.00000 |
| avmnist | learned_posterior_analytic | direct_risk_regression | auroc | 0.28899 | 5 | 5 | 0.00107 | 0.00193 |
| avmnist | top20_precision | event_prevalence | precision | 0.05597 | 5 | 5 | 0.00096 | 0.00193 |

## 十分位排序

| dataset | safe_rate_spearman | benefit_spearman | nondecreasing_steps |
|---|---|---|---|
| avmnist | 0.69415 | -0.99758 | 8.40000 |
| cremad | 1.00000 | 0.98788 | 9.00000 |
| mosei | 0.99757 | 0.14667 | 9.00000 |
| mosi | 0.98485 | 0.60000 | 8.20000 |

原生分数只在其自身模型定义的删除事件上评价，不与同一骨干表作直接显著性排名。