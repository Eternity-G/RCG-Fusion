"""I2-1: audit analytic contribution identification under two fair tracks.

Figure contract
---------------
Core conclusion: posterior integration identifies beneficial coalitions better
than reliability proxies and direct realized-loss regression, while native
scores are diagnosed only inside their own models.
Archetype: quantitative grid with same-backbone AUROC as the hero panel.
Export: Python/Matplotlib, PNG only, 300 dpi, opaque white background.
Review risk: never rank native-score AUROC against same-backbone AUROC as if
their event labels came from one model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/"results"
FIGURES = ROOT/"figures"
DATASETS = ("mosi", "mosei", "cremad", "avmnist")
METHODS = ("max_confidence", "direct_risk_regression",
           "full_posterior_analytic", "plain_mlp_posterior",
           "learned_posterior_analytic")
LABELS = {"max_confidence": "Max probability",
          "direct_risk_regression": "Direct regression",
          "full_posterior_analytic": "Full-output analytic",
          "plain_mlp_posterior": "MLP posterior analytic",
          "learned_posterior_analytic": "RCG analytic"}
COLORS = {"max_confidence": "#9E9E9E", "direct_risk_regression": "#E69F00",
          "full_posterior_analytic": "#8CBBD9", "plain_mlp_posterior": "#5B8DB8",
          "learned_posterior_analytic": "#0072B2"}


def holm(values):
    result = np.ones(len(values)); order = np.argsort(values); running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(values)-rank)*values[index]))
        result[index] = running
    return result


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    def fmt(value):
        return f"{value:.5f}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(fmt(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join((header, rule, *rows))


def paired_test(data, dataset, second, first, metric, alternative="greater"):
    part = data[data.dataset == dataset].pivot(
        index="train_seed", columns="method", values=metric)
    difference = part[second]-part[first]
    pvalue = 1. if np.allclose(difference, 0) else float(
        ttest_1samp(difference, 0, alternative=alternative).pvalue)
    return {"dataset": dataset, "second": second, "first": first, "metric": metric,
            "difference": float(difference.mean()),
            "improved_seeds": int((difference > 0).sum()), "n_seeds": len(difference),
            "paired_t_one_sided_p": pvalue}


def build_statistics(data, deciles):
    primary = []
    for dataset in DATASETS:
        primary.append(paired_test(data, dataset, "learned_posterior_analytic",
                                   "direct_risk_regression", "auroc"))
        part = data[(data.dataset == dataset) &
                    (data.method == "learned_posterior_analytic")].copy()
        difference = part.top20_precision-part.safe_prevalence
        pvalue = float(ttest_1samp(difference, 0, alternative="greater").pvalue)
        primary.append({"dataset": dataset, "second": "top20_precision",
                        "first": "event_prevalence", "metric": "precision",
                        "difference": float(difference.mean()),
                        "improved_seeds": int((difference > 0).sum()),
                        "n_seeds": len(difference), "paired_t_one_sided_p": pvalue})
    primary = pd.DataFrame(primary)
    primary["holm_p"] = holm(primary.paired_t_one_sided_p.to_numpy())

    monotonic = []
    for (dataset, seed), part in deciles.groupby(["dataset", "train_seed"]):
        part = part.sort_values("decile")
        safe_rho = float(spearmanr(part.decile, part.observed_safe_rate).statistic)
        benefit_rho = float(spearmanr(part.decile, part.mean_observed_benefit).statistic)
        monotonic.append({"dataset": dataset, "train_seed": seed,
                          "safe_rate_spearman": safe_rho,
                          "benefit_spearman": benefit_rho,
                          "nondecreasing_safe_steps": int(
                              (np.diff(part.observed_safe_rate) >= -1e-12).sum()),
                          "total_steps": 9})
    return primary, pd.DataFrame(monotonic)


def plot(data, deciles, native, path):
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .8,
                         "legend.frameon": False})
    fig = plt.figure(figsize=(11.8, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.2, 1.2, 1.0))
    ax_a = fig.add_subplot(grid[0, :2]); ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, :2]); ax_d = fig.add_subplot(grid[1, 2])
    x = np.arange(len(DATASETS)); width = .15
    for index, method in enumerate(METHODS):
        part = data[data.method == method].groupby("dataset").auroc.agg(["mean", "std"])
        ax_a.bar(x+(index-2)*width, part.loc[list(DATASETS), "mean"], width,
                 yerr=part.loc[list(DATASETS), "std"], capsize=2,
                 color=COLORS[method], label=LABELS[method])
    names = [value.upper() if value != "cremad" else "CREMA-D" for value in DATASETS]
    ax_a.set_xticks(x, names); ax_a.set_ylim(.3, 1.02); ax_a.axhline(.5, color="#777777", ls="--", lw=.8)
    ax_a.set_ylabel("Beneficial-event AUROC")
    ax_a.set_title("a  Same-backbone contribution identification", loc="left", fontweight="bold")
    ax_a.legend(ncol=3, fontsize=7, loc="upper left")

    learned = data[data.method == "learned_posterior_analytic"].groupby("dataset").agg(
        top20=("top20_precision", "mean"), top20_sd=("top20_precision", "std"),
        prevalence=("safe_prevalence", "mean"), prevalence_sd=("safe_prevalence", "std"))
    ax_b.bar(x-.18, learned.loc[list(DATASETS), "prevalence"], .36,
             yerr=learned.loc[list(DATASETS), "prevalence_sd"], capsize=2,
             color="#BDBDBD", label="Event prevalence")
    ax_b.bar(x+.18, learned.loc[list(DATASETS), "top20"], .36,
             yerr=learned.loc[list(DATASETS), "top20_sd"], capsize=2,
             color="#0072B2", label="Top-20% precision")
    ax_b.set_xticks(x, names, rotation=25, ha="right"); ax_b.set_ylim(0, .75)
    ax_b.set_ylabel("Fraction beneficial")
    ax_b.set_title("b  High-score enrichment", loc="left", fontweight="bold")
    ax_b.legend(fontsize=7)

    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for color, dataset, name in zip(palette, DATASETS, names):
        part = deciles[deciles.dataset == dataset].groupby("decile").agg(
            predicted=("mean_predicted_probability", "mean"),
            observed=("observed_safe_rate", "mean"),
            observed_sd=("observed_safe_rate", "std"))
        ax_c.plot(part.predicted, part.observed, marker="o", ms=3.5,
                  color=color, label=name)
    ax_c.plot([0, 1], [0, 1], color="#777777", ls="--", lw=.8, label="Ideal")
    ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1)
    ax_c.set_xlabel("Predicted beneficial probability")
    ax_c.set_ylabel("Observed beneficial rate")
    ax_c.set_title("c  Probability-decile calibration", loc="left", fontweight="bold")
    ax_c.legend(ncol=3, fontsize=7)

    native_order = ("concat", "tmc", "qmf", "pdf", "i2moe")
    native_labels = ("Max prob.", "TMC", "QMF", "PDF", "I²MoE")
    position = np.arange(len(native_order)); width = .23
    for index, dataset in enumerate(("mosi", "mosei", "cremad")):
        part = native[native.dataset == dataset].set_index("method").loc[list(native_order)]
        ax_d.bar(position+(index-1)*width, part.auroc_mean, width,
                 yerr=part.auroc_std, capsize=2,
                 color=palette[index], label=names[index])
    ax_d.axhline(.5, color="#777777", ls="--", lw=.8)
    ax_d.set_xticks(position, native_labels, rotation=30, ha="right")
    ax_d.set_ylim(.35, .72); ax_d.set_ylabel("Native-score AUROC")
    ax_d.set_title("d  Within-model native-score audit", loc="left", fontweight="bold")
    ax_d.legend(fontsize=7)

    for axis in (ax_a, ax_b, ax_c, ax_d):
        axis.grid(axis="y", color="#E0E0E0", linewidth=.6, zorder=0)
        axis.set_axisbelow(True)
    fig.savefig(path, dpi=300, facecolor="white", transparent=False,
                bbox_inches="tight")
    plt.close(fig)


def main():
    data = pd.read_csv(RESULTS/"e3_contribution_identification_by_seed.csv")
    deciles = pd.read_csv(RESULTS/"e3_analytic_probability_deciles.csv")
    native = pd.read_csv(RESULTS/"e3_native_score_summary.csv")
    primary, monotonic = build_statistics(data, deciles)
    primary.to_csv(RESULTS/"i2_1_primary_comparisons.csv", index=False)
    monotonic.to_csv(RESULTS/"i2_1_decile_monotonicity.csv", index=False)

    summary = data[data.method.isin(METHODS)].groupby(["dataset", "method"]).agg(
        auroc=("auroc", "mean"), auprc=("auprc", "mean"),
        spearman=("spearman", "mean"), top20_precision=("top20_precision", "mean"),
        brier=("brier", "mean"), ece=("ece", "mean"),
        safe_prevalence=("safe_prevalence", "mean")).reset_index()
    summary.to_csv(RESULTS/"i2_1_same_backbone_summary.csv", index=False)
    plot(data, deciles, native, FIGURES/"i2_1_contribution_identification.png")

    direct = primary[(primary["first"] == "direct_risk_regression") &
                     (primary["metric"] == "auroc")]
    enrichment = primary[primary["first"] == "event_prevalence"]
    monotonic_summary = monotonic.groupby("dataset").agg(
        safe_rate_spearman=("safe_rate_spearman", "mean"),
        benefit_spearman=("benefit_spearman", "mean"),
        nondecreasing_steps=("nondecreasing_safe_steps", "mean")).reset_index()
    decision = pd.DataFrame({
        "criterion": ["AUROC gain > 0.05", "5/5 seeds over direct regression",
                      "Top-20 precision above prevalence after Holm",
                      "positive decile safe-rate ordering"],
        "datasets_passed": [int((direct.difference > .05).sum()),
                            int((direct.improved_seeds == 5).sum()),
                            int((enrichment.holm_p < .05).sum()),
                            int((monotonic_summary.safe_rate_spearman > 0).sum())],
        "required": [3, 3, 3, 4],
    })
    decision["pass"] = decision.datasets_passed >= decision.required
    decision.to_csv(RESULTS/"i2_1_decision.csv", index=False)
    report = ["# I2-1 解析贡献识别主实验", "",
              "同一骨干比较与各方法原生分数诊断保持为两个独立轨道。", "",
              "## 预注册判定", "", markdown_table(decision), "",
              "## 主要配对比较", "", markdown_table(primary), "",
              "## 十分位排序", "", markdown_table(monotonic_summary), "",
              "原生分数只在其自身模型定义的删除事件上评价，不与同一骨干表作直接显著性排名。"]
    (RESULTS/"i2_1_report.md").write_text("\n".join(report), encoding="utf-8")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
