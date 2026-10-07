"""I2-3 analysis for fixed-target context interventions.

Figure contract
---------------
Core conclusion: an unchanged target modality has invariant unimodal
reliability, while its realized contribution changes with the other modalities;
posterior-integrated contribution and the listwise route respond to that change.
Archetype: same-backbone mechanism panels plus a separate within-model native
score audit.  These two tracks must not be pooled into one leaderboard.
Export: Python/Matplotlib, PNG only, 300 dpi, opaque white background.
Review risk: a score being context-sensitive is insufficient; sensitivity must
track the direction or endpoint of true conditional contribution.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATASETS = ("mosi", "mosei", "cremad", "avmnist")
TAU = .01
SAME_METHODS = ("max_probability", "negative_entropy", "analytic_expected",
                "analytic_probability", "anchored_listwise")
NATIVE_METHODS = ("concat", "tmc", "qmf", "pdf", "i2moe")
LABELS = {
    "max_probability": "Max probability", "negative_entropy": "Negative entropy",
    "analytic_expected": "Analytic expected", "analytic_probability": "Analytic probability",
    "anchored_listwise": "Anchored listwise", "concat": "Max probability",
    "tmc": "TMC", "qmf": "QMF", "pdf": "PDF", "i2moe": "I²MoE",
}
COLORS = {"max_probability": "#9E9E9E", "negative_entropy": "#D0D0D0",
          "analytic_expected": "#0072B2", "analytic_probability": "#56B4E9",
          "anchored_listwise": "#009E73", "concat": "#9E9E9E",
          "tmc": "#E69F00", "qmf": "#56B4E9", "pdf": "#CC79A7",
          "i2moe": "#009E73"}


def safe_spearman(x, y):
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    value = spearmanr(x, y).statistic
    return 0.0 if not np.isfinite(value) else float(value)


def safe_auc(label, score):
    label = np.asarray(label); score = np.asarray(score)
    if len(label) < 2 or len(np.unique(label)) < 2 or np.allclose(score, score[0]):
        return .5
    return float(roc_auc_score(label, score))


def paired_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["sample_id", "target_modality"]
    clean = frame[frame.corruption_level == 0][keys + ["score", "true_contribution"]]
    clean = clean.rename(columns={"score": "score_clean",
                                  "true_contribution": "contribution_clean"})
    changed = frame[frame.corruption_level > 0].merge(clean, on=keys, validate="many_to_one")
    changed["delta_score"] = changed.score - changed.score_clean
    changed["delta_contribution"] = (
        changed.true_contribution - changed.contribution_clean)
    changed["strict_switch"] = (
        ((changed.contribution_clean > TAU) & (changed.true_contribution < -TAU)) |
        ((changed.contribution_clean < -TAU) & (changed.true_contribution > TAU)))
    return changed


def summarize_group(frame: pd.DataFrame) -> dict[str, float]:
    changed = paired_frame(frame)
    informative = changed[np.abs(changed.delta_contribution) > TAU]
    endpoint = changed.true_contribution > TAU
    if len(informative):
        direction = np.mean(np.sign(informative.delta_score) ==
                            np.sign(informative.delta_contribution))
        delta_auc = safe_auc(informative.delta_contribution > 0,
                             informative.delta_score)
    else:
        direction, delta_auc = np.nan, np.nan
    switched = changed[changed.strict_switch]
    switch_auc = safe_auc(switched.true_contribution > TAU, switched.score)
    return {
        "n_changed": len(changed),
        "score_context_mad": float(np.abs(changed.delta_score).mean()),
        "score_context_max": float(np.abs(changed.delta_score).max()),
        "contribution_context_mad": float(np.abs(changed.delta_contribution).mean()),
        "delta_spearman": safe_spearman(changed.delta_score, changed.delta_contribution),
        "delta_direction_accuracy": float(direction),
        "delta_auroc": float(delta_auc),
        "endpoint_auroc": safe_auc(endpoint, changed.score),
        "strict_switch_rate": float(changed.strict_switch.mean()),
        "strict_switch_count": int(changed.strict_switch.sum()),
        "strict_switch_endpoint_auroc": switch_auc,
    }


def load_and_summarize() -> tuple[pd.DataFrame, dict]:
    buckets = defaultdict(list); trajectory_source = {}
    for dataset in DATASETS:
        root = ROOT / "runs" / f"formal-i2-3-{dataset}"
        for path in (root / "rcg").glob("fold_*/seed_*/target_*.parquet"):
            frame = pd.read_parquet(path)
            for method, part in frame.groupby("method"):
                seed = int(part.train_seed.iloc[0])
                buckets[(dataset, "same_backbone", method, seed)].append(part)
                if dataset == "mosi" and seed == 11 and method in (
                        "max_probability", "analytic_expected", "anchored_listwise"):
                    target = str(part.target_modality.iloc[0])
                    trajectory_source[(method, target)] = part
        for path in (root / "native").glob("*/fold_*/seed_*/target_*.parquet"):
            frame = pd.read_parquet(path)
            method = str(frame.method.iloc[0]); seed = int(frame.train_seed.iloc[0])
            buckets[(dataset, "native_within_model", method, seed)].append(frame)
    rows = []
    for (dataset, track, method, seed), parts in buckets.items():
        value = pd.concat(parts, ignore_index=True)
        rows.append({"dataset": dataset, "track": track, "method": method,
                     "train_seed": seed, **summarize_group(value)})
    return pd.DataFrame(rows), trajectory_source


def holm(values):
    result = np.ones(len(values)); order = np.argsort(values); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values)-rank)*values[index]))
        result[index] = running
    return result


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    same = metrics[metrics.track == "same_backbone"]
    for dataset in DATASETS:
        part = same[same.dataset == dataset].pivot(
            index="train_seed", columns="method", values="delta_spearman")
        for method in ("analytic_expected", "analytic_probability", "anchored_listwise"):
            reference = part[["max_probability", "negative_entropy"]].max(axis=1)
            difference = part[method] - reference
            pvalue = 1.0 if np.allclose(difference, 0) else float(
                ttest_1samp(difference, 0, alternative="greater").pvalue)
            rows.append({"dataset": dataset, "method": method,
                         "reference": "best_unimodal_reliability",
                         "metric": "delta_spearman", "difference": difference.mean(),
                         "improved_seeds": int((difference > 0).sum()),
                         "p": pvalue})
    result = pd.DataFrame(rows)
    result["holm_p"] = holm(result.p.to_numpy())
    return result


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    def fmt(value):
        return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join((header, rule, *rows))


def plot(metrics: pd.DataFrame, path: Path):
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0), constrained_layout=True)
    names = ["MOSI", "MOSEI", "CREMA-D", "AV-MNIST"]
    x = np.arange(len(DATASETS)); width = .16
    same = metrics[metrics.track == "same_backbone"]
    specs = (("score_context_mad", "Mean |score change|", "a  Response to changed context"),
             ("delta_spearman", "Spearman of paired changes", "b  Tracking contribution change"),
             ("delta_auroc", "Paired change AUROC", "c  Direction of contribution change"))
    for axis, (metric, ylabel, title) in zip(axes.flat[:3], specs):
        summary = same.groupby(["dataset", "method"])[metric].agg(["mean", "std"])
        for index, method in enumerate(SAME_METHODS):
            mean = [summary.loc[(dataset, method), "mean"] for dataset in DATASETS]
            std = [summary.loc[(dataset, method), "std"] for dataset in DATASETS]
            axis.bar(x+(index-2)*width, mean, width, yerr=std, capsize=1.5,
                     color=COLORS[method], label=LABELS[method], zorder=3)
        axis.set_xticks(x, names, rotation=15, ha="right")
        axis.set_ylabel(ylabel); axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#E0E0E0", linewidth=.6, zorder=0)
    axes[0, 0].legend(ncol=3, fontsize=7)
    native = metrics[metrics.track == "native_within_model"]
    native_datasets = ("mosi", "mosei", "cremad")
    native_names = ("MOSI", "MOSEI", "CREMA-D")
    nx = np.arange(3); nwidth = .16
    summary = native.groupby(["dataset", "method"]).delta_spearman.agg(["mean", "std"])
    for index, method in enumerate(NATIVE_METHODS):
        mean = [summary.loc[(dataset, method), "mean"] for dataset in native_datasets]
        std = [summary.loc[(dataset, method), "std"] for dataset in native_datasets]
        axes[1, 1].bar(nx+(index-2)*nwidth, mean, nwidth, yerr=std, capsize=1.5,
                       color=COLORS[method], label=LABELS[method], zorder=3)
    axes[1, 1].set_xticks(nx, native_names, rotation=15, ha="right")
    axes[1, 1].set_ylabel("Spearman of paired changes")
    axes[1, 1].set_title("d  Native-score audit within each model", loc="left", fontweight="bold")
    axes[1, 1].grid(axis="y", color="#E0E0E0", linewidth=.6, zorder=0)
    axes[1, 1].legend(ncol=3, fontsize=7)
    fig.savefig(path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


def trajectory_plot(sources: dict, path: Path):
    # Deterministic examples: three MOSI samples with the largest contribution
    # range for seed 11, target audio, chosen before looking at score agreement.
    reference = sources[("analytic_expected", "audio")]
    ranges = reference.groupby("sample_id").true_contribution.agg(lambda x: x.max()-x.min())
    selected = ranges.nlargest(3).index.tolist()
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.2), constrained_layout=True)
    for axis, sample in zip(axes, selected):
        base = reference[reference.sample_id == sample].groupby("corruption_level").agg(
            contribution=("true_contribution", "mean"), score=("score", "mean"))
        axis.plot(base.index, base.contribution, marker="o", color="#111111",
                  label="True contribution")
        axis.plot(base.index, base.score, marker="s", color="#0072B2",
                  label="Analytic expected")
        router = sources[("anchored_listwise", "audio")]
        router = router[router.sample_id == sample].groupby("corruption_level").score.mean()
        axis.plot(router.index, router, marker="^", color="#009E73", label="Listwise score")
        axis.axhline(0, color="#888888", ls="--", lw=.8)
        axis.set_xlabel("Context noise level"); axis.set_ylabel("Contribution / score")
        axis.set_title(str(sample).replace("$", "\\$"), fontsize=8)
        axis.grid(axis="y", color="#E0E0E0", linewidth=.6)
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


def main():
    metrics, trajectories = load_and_summarize()
    metrics.to_csv(RESULTS / "i2_3_context_by_seed.csv", index=False)
    summary = metrics.groupby(["dataset", "track", "method"]).agg(
        score_context_mad=("score_context_mad", "mean"),
        score_context_max=("score_context_max", "max"),
        contribution_context_mad=("contribution_context_mad", "mean"),
        delta_spearman=("delta_spearman", "mean"),
        delta_direction_accuracy=("delta_direction_accuracy", "mean"),
        delta_auroc=("delta_auroc", "mean"), endpoint_auroc=("endpoint_auroc", "mean"),
        strict_switch_rate=("strict_switch_rate", "mean"),
        strict_switch_count=("strict_switch_count", "sum"),
        strict_switch_endpoint_auroc=("strict_switch_endpoint_auroc", "mean"),
    ).reset_index()
    summary.to_csv(RESULTS / "i2_3_context_summary.csv", index=False)
    tests = paired_tests(metrics); tests.to_csv(
        RESULTS / "i2_3_primary_comparisons.csv", index=False)
    same = summary[summary.track == "same_backbone"]
    reliability = same[same.method.isin(("max_probability", "negative_entropy"))]
    analytic = same[same.method == "analytic_expected"].set_index("dataset")
    best_rel = reliability.groupby("dataset").delta_spearman.max()
    decisions = pd.DataFrame([
        {"criterion": "fixed-target reliability is invariant (max change <1e-6)",
         "datasets_passed": int((reliability.groupby("dataset").score_context_max.max() < 1e-6).sum()),
         "required": 4},
        {"criterion": "analytic delta-Spearman exceeds reliability by >0.10",
         "datasets_passed": int(((analytic.delta_spearman-best_rel) > .10).sum()),
         "required": 3},
        {"criterion": "analytic paired-change AUROC >0.65",
         "datasets_passed": int((analytic.delta_auroc > .65).sum()), "required": 3},
        {"criterion": "strict contribution switches observed",
         "datasets_passed": int((analytic.strict_switch_count > 0).sum()), "required": 3},
    ])
    decisions["pass"] = decisions.datasets_passed >= decisions.required
    decisions.to_csv(RESULTS / "i2_3_decision.csv", index=False)
    plot(metrics, FIGURES / "i2_3_context_dependency.png")
    trajectory_plot(trajectories, FIGURES / "i2_3_context_trajectories.png")
    report = ["# I2-3 上下文依赖机制实验", "",
              "目标模态表示保持逐元素不变，只对其余模态施加共享的特征空间高斯扰动。", "",
              "## 预注册判定", "", markdown_table(decisions), "",
              "## 同一骨干摘要", "", markdown_table(same), "",
              "## 配对比较", "", markdown_table(tests), "",
              "外部原生分数只在各自模型的贡献事件上作内部诊断，不与同一骨干轨道直接排名。"]
    (RESULTS / "i2_3_report.md").write_text("\n".join(report), encoding="utf-8")
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
