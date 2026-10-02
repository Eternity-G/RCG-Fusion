"""Aggregate E6 shrinkage policies and draw test benefit-harm Pareto curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"MOSI": "mosi", "MOSEI": "mosei", "CREMA-D": "cremad",
            "AV-MNIST": "avmnist"}
CONTROLLERS = ["fixed", "max_confidence", "negative_entropy",
               "analytic_pi_g1", "analytic_pi_g2", "analytic_pi_g3"]
LABELS = {"fixed": "Fixed", "max_confidence": "Max probability",
          "negative_entropy": "Negative entropy", "analytic_pi_g1": r"$\pi$",
          "analytic_pi_g2": r"$\pi^2$", "analytic_pi_g3": r"$\pi^3$"}
COLORS = {"fixed": "#8C8C8C", "max_confidence": "#D6A657",
          "negative_entropy": "#3E8E7E", "analytic_pi_g1": "#77A6B6",
          "analytic_pi_g2": "#4C78A8", "analytic_pi_g3": "#A98BC3"}


def ci95(values):
    values = np.asarray(values, float)
    return float(stats.t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))


def holm(values):
    order = np.argsort(values); adjusted = np.empty(len(values)); running = 0.
    for rank, index in enumerate(order):
        candidate = min(1., (len(values)-rank)*values[index])
        running = max(running, candidate); adjusted[index] = running
    return adjusted


def aggregate_folds(frame, keys):
    rows = []
    excluded = set(keys) | {"fold", "train_seed", "n_test", "dataset"}
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in excluded]
    for group, values in frame.groupby([*keys, "train_seed"]):
        group = group if isinstance(group, tuple) else (group,)
        weights = values.n_test.to_numpy(float)
        row = dict(zip([*keys, "train_seed"], group)); row["n_test"] = int(weights.sum())
        row.update({metric: float(np.average(values[metric], weights=weights))
                    for metric in numeric})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    deployments, curves = [], []
    for dataset, slug in DATASETS.items():
        deployment = pd.read_csv(ROOT/f"runs/formal-e6-{slug}/metrics_by_seed.csv")
        deployment["dataset"] = dataset; deployments.append(deployment)
        curve = pd.read_csv(ROOT/f"runs/formal-e6-{slug}/curves_by_fold_seed.csv")
        curve["dataset"] = dataset
        curves.append(aggregate_folds(curve, ["controller", "strength"]).assign(dataset=dataset))
    deployment = pd.concat(deployments, ignore_index=True)
    curve_seed = pd.concat(curves, ignore_index=True)
    result_dir, figure_dir = ROOT/"results", ROOT/"figures"
    result_dir.mkdir(exist_ok=True); figure_dir.mkdir(exist_ok=True)
    deployment.to_csv(result_dir/"e6_shrinkage_by_seed.csv", index=False)
    curve_seed.to_csv(result_dir/"e6_shrinkage_curve_by_seed.csv", index=False)

    metrics = ["strength", "mean_alpha", "accuracy_gain", "macro_f1_gain", "nll_gain",
               "relative_nll_reduction", "negative_flip_rate", "correction_rate",
               "clipped_harm", "harmful_action_rate", "regret_reduction", "activation_rate"]
    rows = []
    for (dataset, variant), values in deployment.groupby(["dataset", "variant"]):
        row = {"dataset": dataset, "variant": variant, "n_seeds": len(values)}
        for metric in metrics:
            row[f"{metric}_mean"] = values[metric].mean()
            row[f"{metric}_sd"] = values[metric].std(ddof=1)
            row[f"{metric}_ci95"] = ci95(values[metric].to_numpy())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(result_dir/"e6_shrinkage_summary.csv", index=False)

    curve_rows = []
    for (dataset, controller, strength), values in curve_seed.groupby(
            ["dataset", "controller", "strength"]):
        curve_rows.append({"dataset": dataset, "controller": controller,
                           "strength": strength,
                           "nll_gain_mean": values.nll_gain.mean(),
                           "nll_gain_ci95": ci95(values.nll_gain.to_numpy()),
                           "accuracy_gain_mean": values.accuracy_gain.mean(),
                           "clipped_harm_mean": values.clipped_harm.mean(),
                           "clipped_harm_ci95": ci95(values.clipped_harm.to_numpy()),
                           "mean_alpha_mean": values.mean_alpha.mean()})
    curve_summary = pd.DataFrame(curve_rows)
    curve_summary.to_csv(result_dir/"e6_shrinkage_curve_summary.csv", index=False)

    tests = []
    comparisons = [("analytic_pi_g2", "full_coalition", "pi2_vs_full"),
                   ("analytic_pi_g2", "global_fixed", "pi2_vs_fixed"),
                   ("analytic_pi_g2", "max_confidence", "pi2_vs_confidence"),
                   ("analytic_pi_g2", "negative_entropy", "pi2_vs_entropy"),
                   ("pi2_fixed075", "analytic_pi_g2", "fixed075_vs_selected_pi2")]
    for dataset in DATASETS:
        subset = deployment[deployment.dataset == dataset]
        for left, right, comparison in comparisons:
            for metric in ("nll_gain", "accuracy_gain", "clipped_harm"):
                a = subset[subset.variant == left].sort_values("train_seed")[metric].to_numpy()
                b = subset[subset.variant == right].sort_values("train_seed")[metric].to_numpy()
                difference = a-b; test = stats.ttest_rel(a, b)
                tests.append({"dataset": dataset, "comparison": comparison, "metric": metric,
                              "mean_difference": difference.mean(), "ci95": ci95(difference),
                              "t_statistic": test.statistic, "p_value": test.pvalue})
    adjusted = holm([row["p_value"] for row in tests])
    for row, value in zip(tests, adjusted): row["holm_p"] = value
    pd.DataFrame(tests).to_csv(result_dir/"e6_shrinkage_paired_tests.csv", index=False)

    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7,
                         "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    for label, (dataset, ax) in zip("abcd", zip(DATASETS, axes.flat)):
        part = curve_summary[curve_summary.dataset == dataset]
        for controller in CONTROLLERS:
            values = part[part.controller == controller].sort_values("strength")
            ax.plot(values.clipped_harm_mean*100, values.nll_gain_mean,
                    marker="o", markersize=3, linewidth=1.2,
                    color=COLORS[controller], label=LABELS[controller])
        selected = summary[(summary.dataset == dataset) &
                           (summary.variant == "analytic_pi_g2")].iloc[0]
        fixed = summary[(summary.dataset == dataset) &
                        (summary.variant == "pi2_fixed075")].iloc[0]
        ax.scatter(selected.clipped_harm_mean*100, selected.nll_gain_mean,
                   marker="*", s=55, color="#1F4E79", edgecolor="white", linewidth=.5,
                   zorder=5, label=r"selected $\pi^2$")
        ax.scatter(fixed.clipped_harm_mean*100, fixed.nll_gain_mean,
                   marker="X", s=35, color="#D95F5F", edgecolor="white", linewidth=.4,
                   zorder=5, label=r"$0.75\pi^2$")
        ax.axvline(2, color="#C44E52", linestyle="--", linewidth=.9)
        ax.axhline(0, color="#444444", linewidth=.6)
        ax.set_xlabel("Mean clipped harm (%)"); ax.set_ylabel("NLL improvement")
        ax.set_title(dataset, loc="left", fontweight="bold")
        ax.grid(alpha=.18, linewidth=.5)
        ax.text(-.13, 1.06, label, transform=ax.transAxes,
                fontweight="bold", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="outside upper center",
               ncol=4, bbox_to_anchor=(.5, 1.04))
    fig.suptitle("Contribution-probability shrinkage preserves gains under a 2% harm budget",
                 y=1.075, fontsize=9, fontweight="bold")
    fig.savefig(figure_dir/"e6_adaptive_shrinkage.png", dpi=300,
                facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    main()
