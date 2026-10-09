"""Summarize S1 clean performance and render the manuscript comparison plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT/"results"
PRIMARY = ("coalition_dropout", "concat", "tmc", "qmf", "pdf", "i2moe")
ORDER = ("mosi", "mosei", "cremad", "avmnist")
LABEL = {"mosi": "MOSI", "mosei": "MOSEI", "cremad": "CREMA-D", "avmnist": "AV-MNIST"}


def markdown(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.select_dtypes(include=[np.number]):
        values[column] = values[column].map(lambda value: f"{value:.6f}" if isinstance(value, float) else str(value))
    header = "| " + " | ".join(values.columns) + " |"
    rule = "|" + "|".join("---" for _ in values.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def main():
    metrics = pd.read_csv(RESULT/"s1_clean_metrics.csv")
    paired = pd.read_csv(RESULT/"s1_clean_paired_bootstrap.csv")
    rows = []
    for dataset in ORDER:
        data = metrics[metrics.dataset == dataset]
        rcg = data[data.method == "rcg_fusion_a8_v2"].iloc[0]
        external = data[data.method.isin(PRIMARY)]
        nll_best = external.loc[external.nll.idxmin()]
        accuracy_best = external.loc[external.accuracy.idxmax()]
        nll_test = paired[(paired.dataset == dataset) & (paired.baseline == nll_best.method)].iloc[0]
        accuracy_test = paired[(paired.dataset == dataset) &
                               (paired.baseline == accuracy_best.method)].iloc[0]
        rows.append({"dataset": dataset, "rcg_accuracy": rcg.accuracy,
                     "best_accuracy_baseline": accuracy_best.method,
                     "best_baseline_accuracy": accuracy_best.accuracy,
                     "accuracy_gain": rcg.accuracy-accuracy_best.accuracy,
                     "accuracy_ci_low": accuracy_test.accuracy_ci_low,
                     "accuracy_ci_high": accuracy_test.accuracy_ci_high,
                     "rcg_nll": rcg.nll, "best_nll_baseline": nll_best.method,
                     "best_baseline_nll": nll_best.nll,
                     "nll_gain": nll_best.nll-rcg.nll,
                     "nll_ci_low": nll_test.nll_ci_low,
                     "nll_ci_high": nll_test.nll_ci_high})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT/"s1_primary_summary.csv", index=False)
    decision = pd.DataFrame([
        {"criterion": "lowest NLL among compute-matched primary methods",
         "datasets_passed": int((summary.nll_gain > 0).sum()), "required": 3,
         "pass": bool((summary.nll_gain > 0).sum() >= 3)},
        {"criterion": "paired NLL CI excludes zero against strongest NLL baseline",
         "datasets_passed": int((summary.nll_ci_low > 0).sum()), "required": 3,
         "pass": bool((summary.nll_ci_low > 0).sum() >= 3)},
        {"criterion": "accuracy improves over strongest accuracy baseline",
         "datasets_passed": int((summary.accuracy_gain > 0).sum()), "required": 2,
         "pass": bool((summary.accuracy_gain > 0).sum() >= 2)},
        {"criterion": "accuracy loss no worse than 0.5pp",
         "datasets_passed": int((summary.accuracy_gain >= -.005).sum()), "required": 4,
         "pass": bool((summary.accuracy_gain >= -.005).all())},
    ])
    decision.to_csv(RESULT/"s1_decision.csv", index=False)

    colors = {"RCG-Fusion": "#0072B2", "Best baseline": "#999999"}
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.3), constrained_layout=True)
    x = np.arange(len(ORDER)); width = .36
    axes[0].bar(x-width/2, summary.best_baseline_nll, width, color=colors["Best baseline"], label="Best primary baseline")
    axes[0].bar(x+width/2, summary.rcg_nll, width, color=colors["RCG-Fusion"], label="RCG-Fusion")
    axes[0].set_ylabel("NLL (lower is better)"); axes[0].set_xticks(x, [LABEL[v] for v in ORDER], rotation=20)
    axes[0].set_title("a  Probability quality", loc="left", fontweight="bold")
    axes[1].bar(x-width/2, 100*summary.best_baseline_accuracy, width, color=colors["Best baseline"])
    axes[1].bar(x+width/2, 100*summary.rcg_accuracy, width, color=colors["RCG-Fusion"])
    axes[1].set_ylabel("Accuracy (%)"); axes[1].set_xticks(x, [LABEL[v] for v in ORDER], rotation=20)
    axes[1].set_title("b  Decision accuracy", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", alpha=.2)
    figure = ROOT/"figures/s1_clean_main.png"
    fig.savefig(figure, dpi=300, facecolor="white", transparent=False); plt.close(fig)

    lines = ["# S1 统一干净主结果", "", "## 与最强计算匹配主基线比较", "",
             markdown(summary), "", "## 预注册判定", "",
             markdown(decision), "",
             "说明：当前结果来自每种方法一个五成员集成点；bootstrap刻画样本或簇不确定性，"
             "不替代独立训练集成重复。"]
    (RESULT/"s1_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(summary.to_string(index=False)); print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
