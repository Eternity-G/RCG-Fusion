# S1 统一干净主结果

## 与最强计算匹配主基线比较

| dataset | rcg_accuracy | best_accuracy_baseline | best_baseline_accuracy | accuracy_gain | accuracy_ci_low | accuracy_ci_high | rcg_nll | best_nll_baseline | best_baseline_nll | nll_gain | nll_ci_low | nll_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mosi | 0.803354 | coalition_dropout | 0.797256 | 0.006098 | -0.015989 | 0.025641 | 0.441787 | coalition_dropout | 0.450446 | 0.008659 | -0.003946 | 0.020612 |
| mosei | 0.843148 | concat | 0.846450 | -0.003302 | -0.009801 | 0.003121 | 0.355486 | concat | 0.355143 | -0.000342 | -0.005894 | 0.005204 |
| cremad | 0.637329 | tmc | 0.650900 | -0.013572 | -0.025755 | -0.001479 | 0.962978 | coalition_dropout | 0.973839 | 0.010861 | -0.004509 | 0.025128 |
| avmnist | 0.993611 | i2moe | 0.993889 | -0.000278 | -0.002500 | 0.001667 | 0.021338 | i2moe | 0.025643 | 0.004305 | -0.002042 | 0.011767 |

## 预注册判定

| criterion | datasets_passed | required | pass |
|---|---|---|---|
| lowest NLL among compute-matched primary methods | 3 | 3 | True |
| paired NLL CI excludes zero against strongest NLL baseline | 0 | 3 | False |
| accuracy improves over strongest accuracy baseline | 1 | 2 | False |
| accuracy loss no worse than 0.5pp | 3 | 4 | False |

说明：当前结果来自每种方法一个五成员集成点；bootstrap刻画样本或簇不确定性，不替代独立训练集成重复。
