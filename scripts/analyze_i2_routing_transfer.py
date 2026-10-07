"""I2-2: test whether listwise routing gains survive the decision pipeline.

Figure contract
---------------
Core conclusion: listwise residual routing expands useful candidate coverage on
the non-saturated sentiment tasks, while analytic anchoring preserves the
strong analytic candidate on saturated tasks; candidate gains only count when
they propagate through the identical mixer and shrinkage controller.
Archetype: four-panel quantitative comparison with candidate recovery and
downstream NLL gain as the two primary panels.
Export: Python/Matplotlib, PNG only, 300 dpi, opaque white background.
Review risk: do not present structural anchor preservation or larger candidate
coverage as a task-performance improvement when downstream gains are absent.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATASETS = ("mosi", "mosei", "cremad", "avmnist")
VARIANTS = (
    "analytic_only",
    "hard_oracle_classifier",
    "soft_listwise",
    "pairwise_only",
    "listwise_pairwise_unanchored",
    "listwise_pairwise_anchored",
)
LABELS = {
    "analytic_only": "Analytic",
    "hard_oracle_classifier": "Pointwise hard",
    "soft_listwise": "Listwise",
    "pairwise_only": "Pairwise",
    "listwise_pairwise_unanchored": "List + pair",
    "listwise_pairwise_anchored": "Anchored list + pair",
}
COLORS = {
    "analytic_only": "#9E9E9E",
    "hard_oracle_classifier": "#E69F00",
    "soft_listwise": "#56B4E9",
    "pairwise_only": "#009E73",
    "listwise_pairwise_unanchored": "#CC79A7",
    "listwise_pairwise_anchored": "#0072B2",
}


def holm(values: np.ndarray) -> np.ndarray:
    result = np.ones(len(values)); order = np.argsort(values); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"

    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value)

    rows = ["| " + " | ".join(fmt(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join((header, rule, *rows))


def load_data() -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        path = ROOT / "runs" / f"formal-i2-2-{dataset}" / "metrics_by_seed.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        current = pd.read_csv(path)
        if set(current.variant) != set(VARIANTS):
            raise RuntimeError(f"Incomplete I2-2 variants for {dataset}")
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def paired(data: pd.DataFrame, dataset: str, second: str, first: str,
           metric: str, *, larger_is_better: bool = True) -> dict[str, object]:
    part = data[data.dataset == dataset].pivot(
        index="train_seed", columns="variant", values=metric)
    difference = part[second] - part[first]
    oriented = difference if larger_is_better else -difference
    pvalue = 1.0 if np.allclose(oriented, 0) else float(
        ttest_1samp(oriented, 0, alternative="greater").pvalue)
    return {
        "dataset": dataset,
        "second": second,
        "first": first,
        "metric": metric,
        "difference_second_minus_first": float(difference.mean()),
        "improved_seeds": int((oriented > 0).sum()),
        "n_seeds": int(len(oriented)),
        "paired_t_one_sided_p": pvalue,
    }


def statistics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    # The two sentiment-task recovery tests are the pre-registered non-saturated
    # comparisons and form one Holm family.
    for dataset in ("mosi", "mosei"):
        row = paired(data, dataset, "listwise_pairwise_anchored", "analytic_only",
                     "route_candidate_oracle_recovery")
        row["family"] = "non_saturated_candidate_recovery"
        rows.append(row)
    # Downstream transfer is a separate four-dataset family.
    for dataset in DATASETS:
        row = paired(data, dataset, "listwise_pairwise_anchored", "analytic_only",
                     "adaptive_nll_gain")
        row["family"] = "downstream_nll_transfer"
        rows.append(row)
    # Anchoring is compared with the same learned score without the structural
    # anchor. Lower action-oracle NLL and higher deployed NLL gain are preferred.
    for dataset in DATASETS:
        row = paired(data, dataset, "listwise_pairwise_anchored",
                     "listwise_pairwise_unanchored", "route_action_oracle_nll",
                     larger_is_better=False)
        row["family"] = "anchoring_action_oracle"
        rows.append(row)
    result = pd.DataFrame(rows)
    result["holm_p"] = 1.0
    for _, indexes in result.groupby("family").groups.items():
        indexes = list(indexes)
        result.loc[indexes, "holm_p"] = holm(
            result.loc[indexes, "paired_t_one_sided_p"].to_numpy())

    pivot = data.pivot_table(index=["dataset", "train_seed"], columns="variant")
    decisions = []
    recovery = result[result.family == "non_saturated_candidate_recovery"]
    decisions.append({
        "criterion": "MOSI/MOSEI candidate recovery gain > 2pp with Holm p<0.05",
        "datasets_passed": int(((recovery.difference_second_minus_first > .02) &
                                (recovery.holm_p < .05)).sum()),
        "required": 2,
    })
    transfer = result[result.family == "downstream_nll_transfer"]
    decisions.append({
        "criterion": "anchored routing improves downstream NLL gain",
        "datasets_passed": int((transfer.difference_second_minus_first > 0).sum()),
        "required": 2,
    })
    saturated = transfer[transfer.dataset.isin(("cremad", "avmnist"))]
    decisions.append({
        "criterion": "saturated tasks lose no more than 0.0005 NLL gain",
        "datasets_passed": int((saturated.difference_second_minus_first >= -.0005).sum()),
        "required": 2,
    })
    anchor_delta = (
        pivot["adaptive_nll_gain"]["listwise_pairwise_anchored"]
        - pivot["adaptive_nll_gain"]["listwise_pairwise_unanchored"]
    ).groupby("dataset").mean()
    decisions.append({
        "criterion": "anchoring does not reduce mean downstream gain by >0.0005",
        "datasets_passed": int((anchor_delta >= -.0005).sum()),
        "required": 4,
    })
    decision = pd.DataFrame(decisions)
    decision["pass"] = decision.datasets_passed >= decision.required
    return result, decision


def plot(data: pd.DataFrame, path: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)
    names = ["MOSI", "MOSEI", "CREMA-D", "AV-MNIST"]
    x = np.arange(len(DATASETS)); width = .125
    metrics = (
        ("route_candidate_oracle_recovery", "Candidate oracle recovery", "a  Useful candidate coverage"),
        ("route_ndcg", "NDCG", "b  Full-list ranking quality"),
        ("adaptive_nll_gain", "NLL reduction vs full fusion", "c  Transfer through mixer + shrinkage"),
        ("adaptive_regret_reduction", "Fraction of fusion regret removed", "d  Decision-level regret reduction"),
    )
    for axis, (metric, ylabel, title) in zip(axes.flat, metrics):
        summary = data.groupby(["dataset", "variant"])[metric].agg(["mean", "std"])
        for index, variant in enumerate(VARIANTS):
            mean = [summary.loc[(dataset, variant), "mean"] for dataset in DATASETS]
            std = [summary.loc[(dataset, variant), "std"] for dataset in DATASETS]
            axis.bar(x + (index - 2.5) * width, mean, width, yerr=std, capsize=1.5,
                     color=COLORS[variant], label=LABELS[variant], zorder=3)
        axis.set_xticks(x, names, rotation=15, ha="right")
        axis.set_ylabel(ylabel); axis.set_title(title, loc="left", fontweight="bold")
        axis.axhline(0, color="#777777", lw=.8)
        axis.grid(axis="y", color="#E0E0E0", linewidth=.6, zorder=0)
    axes[0, 0].legend(ncol=3, fontsize=7, loc="lower right")
    fig.savefig(path, dpi=300, facecolor="white", transparent=False,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    data = load_data()
    data.to_csv(RESULTS / "i2_2_routing_by_seed.csv", index=False)
    summary = data.groupby(["dataset", "variant"]).agg(
        ndcg=("route_ndcg", "mean"),
        pairwise_accuracy=("route_pairwise_accuracy", "mean"),
        candidate_recovery=("route_candidate_oracle_recovery", "mean"),
        candidate_oracle_nll=("route_candidate_oracle_nll", "mean"),
        candidate_headroom_fraction=("route_candidate_headroom_fraction", "mean"),
        adaptive_nll_gain=("adaptive_nll_gain", "mean"),
        adaptive_accuracy_gain=("adaptive_accuracy_gain", "mean"),
        correction_rate=("adaptive_correction_rate", "mean"),
        negative_flip_rate=("adaptive_negative_flip_rate", "mean"),
        clipped_harm=("adaptive_clipped_harm", "mean"),
        regret_reduction=("adaptive_regret_reduction", "mean"),
    ).reset_index()
    summary.to_csv(RESULTS / "i2_2_routing_summary.csv", index=False)
    primary, decision = statistics(data)
    primary.to_csv(RESULTS / "i2_2_primary_comparisons.csv", index=False)
    decision.to_csv(RESULTS / "i2_2_decision.csv", index=False)
    plot(data, FIGURES / "i2_2_routing_transfer.png")

    anchored = summary[summary.variant == "listwise_pairwise_anchored"]
    analytic = summary[summary.variant == "analytic_only"]
    report = [
        "# I2-2 列表式残差路由与下游传递", "",
        "候选排序和最终决策使用相同骨干、联盟概率、后验、mixer及收缩控制器。",
        "列表路由只有在候选覆盖能够传递到决策收益时才计作系统性能证据。", "",
        "## 预注册判定", "", markdown_table(decision), "",
        "## 主要配对比较", "", markdown_table(primary), "",
        "## 锚定列表式路由摘要", "", markdown_table(anchored), "",
        "## 解析排序摘要", "", markdown_table(analytic), "",
        "MOSI/MOSEI属于候选非饱和任务；CREMA-D和AV-MNIST的解析候选覆盖已较高。",
        "结构性Top-1锚定本身不是经验性能结果，正文只将其作为候选保护机制。",
    ]
    (RESULTS / "i2_2_report.md").write_text("\n".join(report), encoding="utf-8")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
