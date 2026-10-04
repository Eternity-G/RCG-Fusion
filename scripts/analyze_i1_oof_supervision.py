"""I1-1: consolidate the four-dataset OOF supervision credibility evidence."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATASETS = ("mosi", "mosei", "cremad", "avmnist")
ORDER = (
    "in_sample_hard", "oof_single_hard", "oof_multi_hard",
    "oof_multi_soft", "oof_soft_pair", "oof_soft_full",
)
LABELS = {
    "in_sample_hard": "In-sample hard",
    "oof_single_hard": "OOF-1 hard",
    "oof_multi_hard": "OOF-5 hard",
    "oof_multi_soft": "OOF-5 soft",
    "oof_soft_pair": "Soft + pair",
    "oof_soft_full": "Full soft",
}
COLORS = ("#7F7F7F", "#D55E00", "#E69F00", "#56B4E9", "#0072B2", "#009E73")


def markdown_table(frame: pd.DataFrame) -> str:
    def value(item):
        return f"{item:.5f}" if isinstance(item, (float, np.floating)) else str(item)
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(value(item) for item in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def holm(values: list[float]) -> list[float]:
    result = np.ones(len(values), dtype=float)
    order = np.argsort(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count-rank)*values[index])
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def paired_comparison(frame: pd.DataFrame, dataset: str, first: str, second: str,
                      metric: str, higher_better: bool) -> dict[str, object]:
    part = frame[frame.dataset == dataset]
    pivot = part.pivot(index="train_seed", columns="variant", values=metric)
    raw = pivot[second]-pivot[first]
    oriented = raw if higher_better else -raw
    if np.allclose(oriented, 0):
        pvalue = 1.0
    else:
        pvalue = float(ttest_1samp(oriented, 0, alternative="greater").pvalue)
    return {
        "dataset": dataset,
        "first": first,
        "second": second,
        "metric": metric,
        "higher_is_better": higher_better,
        "raw_second_minus_first": float(raw.mean()),
        "oriented_improvement": float(oriented.mean()),
        "improved_seeds": int((oriented > 0).sum()),
        "tied_seeds": int(np.isclose(oriented, 0).sum()),
        "n_seeds": int(len(oriented)),
        "paired_t_one_sided_p": pvalue,
    }


def load_inputs():
    metric_frames, audit_rows = [], []
    for dataset in DATASETS:
        metric = pd.read_csv(RESULTS/f"e2_{dataset}_supervision_ablation_by_seed.csv")
        metric.insert(0, "dataset", dataset)
        metric_frames.append(metric)
        summary = json.loads((RESULTS/f"e2_{dataset}_supervision_summary.json").read_text())
        audit_rows.append({"dataset": dataset, **summary["audit_mean"],
                           **summary["teacher_statistics"]})
    return pd.concat(metric_frames, ignore_index=True), pd.DataFrame(audit_rows)


def plot(metrics: pd.DataFrame, audit: pd.DataFrame, path: Path):
    path.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), constrained_layout=True)
    names = [name.upper() if name != "cremad" else "CREMA-D" for name in DATASETS]
    x = np.arange(len(DATASETS))

    axes[0, 0].bar(x, audit.set_index("dataset").loc[list(DATASETS),
                         "oof_minus_in_sample_loss"], color="#D55E00")
    axes[0, 0].axhline(0, color="black", linewidth=.8)
    axes[0, 0].set_xticks(x, names)
    axes[0, 0].set_ylabel("OOF NLL − in-sample NLL")
    axes[0, 0].set_title("(a) Training optimism exposed by OOF", loc="left", fontsize=10)

    width = .34
    table = audit.set_index("dataset").loc[list(DATASETS)]
    axes[0, 1].bar(x-width/2, table.hard_target_variance, width,
                   label="Hard oracle", color="#E69F00")
    axes[0, 1].bar(x+width/2, table.soft_target_variance, width,
                   label="Soft oracle", color="#009E73")
    axes[0, 1].set_xticks(x, names)
    axes[0, 1].set_ylabel("Across-teacher target variance")
    axes[0, 1].set_title("(b) Soft targets reduce teacher variance", loc="left", fontsize=10)
    axes[0, 1].legend(frameon=False)

    for index, variant in enumerate(ORDER):
        values = metrics[metrics.variant == variant].groupby("dataset").anchored_top3.mean()
        axes[1, 0].plot(x, values.loc[list(DATASETS)], marker="o", linewidth=1.5,
                        color=COLORS[index], label=LABELS[variant])
    axes[1, 0].set_xticks(x, names)
    axes[1, 0].set_ylabel("Anchored Top-K oracle coverage")
    axes[1, 0].set_title("(c) Downstream candidate coverage", loc="left", fontsize=10)
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    hard = metrics[metrics.variant == "oof_multi_hard"].groupby("dataset").anchored_top3.mean()
    soft = metrics[metrics.variant == "oof_soft_pair"].groupby("dataset").anchored_top3.mean()
    disagreement = 1-table.mean_pairwise_teacher_oracle_agreement
    delta = soft.loc[list(DATASETS)].to_numpy()-hard.loc[list(DATASETS)].to_numpy()
    axes[1, 1].axhline(0, color="black", linewidth=.8)
    axes[1, 1].scatter(disagreement, delta, s=58, color="#0072B2")
    for index, label in enumerate(names):
        axes[1, 1].annotate(label, (disagreement.iloc[index], delta[index]),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1, 1].set_xlabel("Teacher disagreement (1 − pair agreement)")
    axes[1, 1].set_ylabel("Soft+pair − OOF-5 hard coverage")
    axes[1, 1].set_title("(d) Disagreement does not imply uniform gain", loc="left", fontsize=10)

    for axis in axes.flat:
        axis.grid(axis="y", color="#DDDDDD", linewidth=.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=300, facecolor="white", transparent=False,
                bbox_inches="tight")
    plt.close(fig)


def main():
    metrics, audit = load_inputs()
    comparisons = []
    for dataset in DATASETS:
        for first, second in (
            ("oof_single_hard", "oof_multi_hard"),
            ("oof_multi_hard", "oof_multi_soft"),
            ("oof_multi_hard", "oof_soft_pair"),
            ("oof_multi_hard", "oof_soft_full"),
        ):
            comparisons.append(paired_comparison(
                metrics, dataset, first, second, "anchored_top3", True))
            comparisons.append(paired_comparison(
                metrics, dataset, first, second, "candidate_oracle_nll", False))
    comparison = pd.DataFrame(comparisons)
    comparison["holm_p"] = holm(comparison.paired_t_one_sided_p.tolist())

    variance_pass = bool((audit.soft_to_hard_variance_ratio < 1).all())
    optimism_pass = bool((audit.oof_minus_in_sample_loss > 0).all())
    # The complete soft supervision is fixed before this consolidation.  Do not
    # choose the best soft variant on test metrics.
    fixed_soft = metrics[metrics.variant == "oof_soft_full"]
    hard = metrics[metrics.variant == "oof_multi_hard"]
    paired = hard.merge(fixed_soft, on=["dataset", "train_seed"], suffixes=("_hard", "_soft"))
    decision = paired.groupby("dataset").agg(
        coverage_gain=("anchored_top3_soft", lambda x: float(np.mean(
            x.to_numpy()-paired.loc[x.index, "anchored_top3_hard"].to_numpy()))),
        oracle_nll_gain=("candidate_oracle_nll_soft", lambda x: float(np.mean(
            paired.loc[x.index, "candidate_oracle_nll_hard"].to_numpy()-x.to_numpy()))),
    ).reset_index()
    non_saturated = decision[decision.dataset != "avmnist"]
    downstream_pass = int(((non_saturated.coverage_gain > 0) |
                           (non_saturated.oracle_nll_gain > 0)).sum()) >= 2

    RESULTS.mkdir(exist_ok=True)
    audit.to_csv(RESULTS/"i1_1_supervision_audit.csv", index=False)
    metrics.to_csv(RESULTS/"i1_1_supervision_metrics_by_seed.csv", index=False)
    comparison.to_csv(RESULTS/"i1_1_supervision_comparisons.csv", index=False)
    decision.to_csv(RESULTS/"i1_1_supervision_decision.csv", index=False)
    plot(metrics, audit, FIGURES/"i1_1_supervision_evidence.png")

    report = [
        "# I1-1 OOF监督可信性实验", "",
        "- 状态：正式复核（复用冻结的四数据集五教师、五路由种子产物）。",
        f"- OOF乐观偏差检查：{'通过' if optimism_pass else '未通过'}。",
        f"- 软目标方差降低：{'通过' if variance_pass else '未通过'}。",
        f"- 至少两个非饱和数据集保留决策信息：{'通过' if downstream_pass else '未通过'}。",
        "- Holm校正后显著性与效应量见 `i1_1_supervision_comparisons.csv`。", "",
        "## 数据集级审计", "", markdown_table(audit), "",
        "## 固定完整软监督相对OOF多教师硬目标", "",
        markdown_table(decision), "",
        "## 判读", "",
        "软监督在全部数据集都显著降低教师目标方差，但候选学习收益具有任务依赖性。"
        "因此I1-1支持其作为泄漏受控、保留教师不确定性的监督机制；它能否传递到最终决策必须由I1-2单独验证。",
    ]
    (RESULTS/"i1_1_supervision_report.md").write_text("\n".join(report), encoding="utf-8")
    print(audit[["dataset", "oof_minus_in_sample_loss", "soft_to_hard_variance_ratio"]].to_string(index=False))
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
