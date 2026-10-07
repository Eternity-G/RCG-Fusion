"""Summarize I3-2 matched-harm decisions and render the paper Pareto figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT/"results", ROOT/"figures"
DATASETS = ("mosi", "mosei", "cremad", "avmnist")
BUDGETS = (.005, .01, .02, .05)
LABELS = {"fixed": "Fixed alpha", "max_confidence": "Max probability",
          "negative_entropy": "Entropy", "analytic_pi_g1": r"$\pi$",
          "analytic_pi_g2": r"$\pi^2$ (ours)", "analytic_pi_g3": r"$\pi^3$",
          "analytic_pi_selected": r"Selection $\gamma$ (ours)",
          "native_tmc": "TMC evidence", "native_qmf": "QMF quality",
          "native_pdf": "PDF Co-Belief"}
COLORS = {"fixed": "#7F7F7F", "max_confidence": "#E69F00",
          "negative_entropy": "#009E73", "analytic_pi_g1": "#56B4E9",
          "analytic_pi_g2": "#D55E00", "analytic_pi_g3": "#CC79A7",
          "analytic_pi_selected": "#D55E00",
          "native_tmc": "#0072B2", "native_qmf": "#F0E442",
          "native_pdf": "#000000"}


def holm(values):
    output = np.ones(len(values)); order = np.argsort(values); running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(values)-rank)*values[index]))
        output[index] = running
    return output


def ci95(values):
    values = np.asarray(values, float)
    if len(values) < 2:
        return 0.
    return float(2.7764451051977987*values.std(ddof=1)/np.sqrt(len(values)))


def load_curves():
    frames = []
    for dataset in DATASETS:
        frame = pd.read_csv(ROOT/"runs"/f"formal-i3-2-{dataset}"/"curves_by_fold_seed.csv")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def choose_at_budgets(curves):
    chosen = []
    for keys, values in curves.groupby(["dataset", "fold", "train_seed", "action", "controller"]):
        for budget in BUDGETS:
            feasible = values[values.selection_clipped_harm <= budget+1e-12]
            if feasible.empty:
                feasible = values.loc[[values.selection_clipped_harm.idxmin()]]
            best_nll = feasible.selection_nll.min()
            row = feasible[feasible.selection_nll <= best_nll+1e-12].sort_values(
                ["test_mean_alpha", "strength"]).iloc[0].to_dict()
            row["budget"] = budget; chosen.append(row)
    # Gamma is a method hyperparameter, selected without test labels from the
    # same analytic family and the same harm constraint.
    analytic = curves[(curves.action == "candidate_mixer") &
                      curves.controller.str.startswith("analytic_pi_g")]
    for keys, values in analytic.groupby(["dataset", "fold", "train_seed", "action"]):
        for budget in BUDGETS:
            feasible = values[values.selection_clipped_harm <= budget+1e-12]
            if feasible.empty:
                feasible = values.loc[[values.selection_clipped_harm.idxmin()]]
            best_nll = feasible.selection_nll.min()
            selected = feasible[feasible.selection_nll <= best_nll+1e-12].sort_values(
                ["test_mean_alpha", "strength"]).iloc[0].to_dict()
            selected["selected_controller"] = selected["controller"]
            selected["controller"] = "analytic_pi_selected"
            selected["budget"] = budget; chosen.append(selected)
    chosen = pd.DataFrame(chosen)
    rows = []
    numeric = [column for column in chosen.select_dtypes(include=[np.number]).columns
               if column not in {"fold", "train_seed", "n_test", "budget"}]
    for keys, values in chosen.groupby(["dataset", "train_seed", "action", "controller", "budget"]):
        weights = values.n_test.to_numpy(float)
        row = dict(zip(["dataset", "train_seed", "action", "controller", "budget"], keys))
        row["n_test"] = int(weights.sum())
        row.update({column: float(np.average(values[column], weights=weights))
                    for column in numeric})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_dense(curves):
    rows = []
    numeric = [column for column in curves.select_dtypes(include=[np.number]).columns
               if column not in {"fold", "train_seed", "n_test", "strength"}]
    for keys, values in curves.groupby(["dataset", "train_seed", "action", "controller", "strength"]):
        weights = values.n_test.to_numpy(float)
        row = dict(zip(["dataset", "train_seed", "action", "controller", "strength"], keys))
        row.update({column: float(np.average(values[column], weights=weights))
                    for column in numeric})
        rows.append(row)
    by_seed = pd.DataFrame(rows)
    summary = by_seed.groupby(["dataset", "action", "controller", "strength"]).agg(
        test_nll_gain_mean=("test_nll_gain", "mean"),
        test_clipped_harm_mean=("test_clipped_harm", "mean"),
        test_accuracy_gain_mean=("test_accuracy_gain", "mean")).reset_index()
    return by_seed, summary


def summarize(chosen):
    metrics = ("test_nll_gain", "test_accuracy_gain", "test_macro_f1_gain",
               "test_clipped_harm", "test_harm_p95", "test_negative_flip_rate",
               "test_correction_rate", "test_regret_reduction", "test_mean_alpha")
    rows = []
    for keys, values in chosen.groupby(["dataset", "action", "controller", "budget"]):
        row = dict(zip(["dataset", "action", "controller", "budget"], keys))
        for metric in metrics:
            row[f"{metric}_mean"] = values[metric].mean()
            row[f"{metric}_ci95"] = ci95(values[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(chosen):
    rows = []
    primary = chosen[(chosen.action == "candidate_mixer") &
                     np.isclose(chosen.budget, .02)]
    for dataset in DATASETS:
        pivot = primary[primary.dataset == dataset].pivot(
            index="train_seed", columns="controller", values="test_nll_gain")
        references = [name for name in ("fixed", "max_confidence", "negative_entropy",
                                        "native_tmc", "native_qmf", "native_pdf")
                      if name in pivot]
        for reference in references:
            difference = pivot.analytic_pi_selected-pivot[reference]
            pvalue = 1. if np.allclose(difference, 0) else float(
                ttest_1samp(difference, 0, alternative="greater").pvalue)
            rows.append({"dataset": dataset, "comparison": f"selected_pi_vs_{reference}",
                         "budget": .02, "nll_gain_difference": difference.mean(),
                         "improved_seeds": int((difference > 0).sum()), "p": pvalue})
    result = pd.DataFrame(rows); result["holm_p"] = holm(result.p.to_numpy())
    return result


def pareto_front(points):
    ordered = points.sort_values("test_clipped_harm_mean")
    best = -np.inf; keep = []
    for index, row in ordered.iterrows():
        if row.test_nll_gain_mean > best+1e-12:
            keep.append(index); best = row.test_nll_gain_mean
    return ordered.loc[keep]


def normalized_hypervolume(points, max_harm=.05):
    values = points[(points.test_clipped_harm_mean >= 0) &
                    (points.test_clipped_harm_mean <= max_harm)].sort_values(
                        "test_clipped_harm_mean")
    x = np.r_[0., values.test_clipped_harm_mean.to_numpy(), max_harm]
    y = np.r_[0., values.test_nll_gain_mean.to_numpy(),
              values.test_nll_gain_mean.max() if len(values) else 0.]
    y = np.maximum.accumulate(np.maximum(y, 0))
    return float(np.trapz(y, x)/max_harm)


def controller_front_membership(summary, budget_summary):
    rows = []
    primary = summary[summary.action == "candidate_mixer"]
    for dataset in DATASETS:
        part = primary[primary.dataset == dataset]
        front = pareto_front(part)
        budget_part = budget_summary[(budget_summary.dataset == dataset) &
                                     (budget_summary.action == "candidate_mixer")]
        for controller in part.controller.unique():
            selected = budget_part[budget_part.controller == controller].sort_values("budget")
            rows.append({"dataset": dataset, "controller": controller,
                         "on_pareto_front": bool((front.controller == controller).any()),
                         "pareto_hypervolume": normalized_hypervolume(
                             part[part.controller == controller]),
                         "budgeted_utility_auc": (float(np.trapz(
                             selected.test_nll_gain_mean, selected.budget)/.045)
                             if len(selected) > 1 else np.nan)})
    return pd.DataFrame(rows)


def matched_test_envelope(dense, budget=.02):
    """Descriptive Pareto envelope at a common observed test-harm budget.

    This table compares trade-off curves; it is not used to choose a deployed
    strength and is therefore kept separate from selection-chosen operating
    points.
    """
    rows = []
    primary = dense[(dense.action == "candidate_mixer") &
                    (dense.test_clipped_harm_mean <= budget+1e-12)]
    for (dataset, controller), values in primary.groupby(["dataset", "controller"]):
        best = values.sort_values(["test_nll_gain_mean", "test_clipped_harm_mean"],
                                  ascending=[False, True]).iloc[0]
        rows.append({"dataset": dataset, "controller": controller, "budget": budget,
                     "strength": best.strength,
                     "test_clipped_harm": best.test_clipped_harm_mean,
                     "test_nll_gain": best.test_nll_gain_mean,
                     "test_accuracy_gain": best.test_accuracy_gain_mean})
    return pd.DataFrame(rows)


def plot(dense, selected, path):
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.2, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.9), constrained_layout=True)
    names = {"mosi": "MOSI", "mosei": "MOSEI", "cremad": "CREMA-D", "avmnist": "AV-MNIST"}
    for letter, dataset, axis in zip("abcd", DATASETS, axes.flat):
        part = dense[(dense.dataset == dataset) & (dense.action == "candidate_mixer")]
        for controller in [name for name in LABELS if name != "analytic_pi_selected"]:
            values = part[part.controller == controller].sort_values("test_clipped_harm_mean")
            if values.empty:
                continue
            axis.plot(100*values.test_clipped_harm_mean, values.test_nll_gain_mean,
                      marker="o", ms=3.2, lw=1.15, color=COLORS[controller],
                      label=LABELS[controller], alpha=.45 if controller.startswith("analytic_pi_g") else .9,
                      zorder=4 if controller == "analytic_pi_g2" else 2)
        operating = selected[(selected.dataset == dataset) &
                             (selected.action == "candidate_mixer") &
                             (selected.controller == "analytic_pi_selected")]
        axis.plot(100*operating.test_clipped_harm_mean, operating.test_nll_gain_mean,
                  color=COLORS["analytic_pi_selected"], marker="*", ms=7, lw=1.8,
                  label=LABELS["analytic_pi_selected"], zorder=6)
        annotation = operating.assign(
            _x=(100*operating.test_clipped_harm_mean).round(7),
            _y=operating.test_nll_gain_mean.round(7))
        for (_, _), values in annotation.groupby(["_x", "_y"]):
            row = values.iloc[0]; budgets = sorted(values.budget*100)
            label = (f"{budgets[0]:g}%" if len(budgets) == 1
                     else f"{budgets[0]:g}–{budgets[-1]:g}%")
            axis.annotate(label, (row._x, row._y),
                          xytext=(2, 2), textcoords="offset points", fontsize=5.8,
                          color=COLORS["analytic_pi_selected"])
        axis.axhline(0, color="#666666", lw=.65); axis.axvline(2, color="#999999", ls="--", lw=.7)
        axis.set_xlabel("Test mean clipped harm (%)"); axis.set_ylabel("Test NLL improvement")
        axis.set_title(names[dataset], loc="left", fontweight="bold")
        axis.grid(color="#E3E3E3", lw=.55); axis.text(-.12, 1.04, letter,
            transform=axis.transAxes, fontweight="bold", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="outside lower center", ncol=5,
               bbox_to_anchor=(.5, -.045), fontsize=6.5)
    fig.suptitle("Matched-harm benefit–risk trade-offs for one frozen candidate action",
                 y=1.025, fontsize=9, fontweight="bold")
    fig.savefig(path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame):
    def value(item):
        return f"{item:.6f}" if isinstance(item, (float, np.floating)) else str(item)
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "|" + "|".join("---" for _ in frame.columns) + "|",
                      *("| " + " | ".join(value(item) for item in row) + " |"
                        for row in frame.itertuples(index=False, name=None))])


def main():
    curves = load_curves(); chosen = choose_at_budgets(curves); summary = summarize(chosen)
    dense_by_seed, dense_summary = aggregate_dense(curves)
    tests = paired_tests(chosen); fronts = controller_front_membership(dense_summary, summary)
    matched = matched_test_envelope(dense_summary)
    curves.to_csv(RESULTS/"i3_2_dense_curves_by_fold_seed.csv", index=False)
    chosen.to_csv(RESULTS/"i3_2_budget_choices_by_seed.csv", index=False)
    summary.to_csv(RESULTS/"i3_2_budget_summary.csv", index=False)
    dense_by_seed.to_csv(RESULTS/"i3_2_dense_curves_by_seed.csv", index=False)
    dense_summary.to_csv(RESULTS/"i3_2_dense_curve_summary.csv", index=False)
    tests.to_csv(RESULTS/"i3_2_primary_comparisons.csv", index=False)
    fronts.to_csv(RESULTS/"i3_2_pareto_membership.csv", index=False)
    matched.to_csv(RESULTS/"i3_2_matched_test_envelope.csv", index=False)
    primary = summary[(summary.action == "candidate_mixer") &
                      (summary.controller == "analytic_pi_selected") &
                      np.isclose(summary.budget, .02)]
    pi_front = fronts[fronts.controller.str.startswith("analytic_pi")].groupby(
        "dataset").on_pareto_front.any()
    matched_pass = 0
    for dataset in DATASETS:
        part = matched[matched.dataset == dataset].set_index("controller")
        analytic = part.loc[[name for name in part.index if name.startswith("analytic_pi_g")],
                            "test_nll_gain"].max()
        references = part.loc[[name for name in ("fixed", "max_confidence", "negative_entropy")
                               if name in part.index], "test_nll_gain"]
        matched_pass += int(len(references) == 3 and analytic > references.max())
    decision = pd.DataFrame([
        {"criterion": "contribution-probability family reaches Pareto front",
         "datasets_passed": int(pi_front.sum()), "required": 3},
        {"criterion": "contribution family beats simple controls at matched 2% test harm",
         "datasets_passed": matched_pass, "required": 3},
        {"criterion": "selection-gamma test harm remains within 2% budget",
         "datasets_passed": int((primary.test_clipped_harm_mean <= .02).sum()), "required": 3},
        {"criterion": "selection-gamma correction rate exceeds negative flips",
         "datasets_passed": int((primary.test_correction_rate_mean >
                                  primary.test_negative_flip_rate_mean).sum()), "required": 4},
        {"criterion": "selection-gamma has no >0.5pp mean accuracy loss",
         "datasets_passed": int((primary.test_accuracy_gain_mean >= -.005).sum()), "required": 4},
    ])
    decision["pass"] = decision.datasets_passed >= decision.required
    decision.to_csv(RESULTS/"i3_2_decision.csv", index=False)
    plot(dense_summary, summary, FIGURES/"i3_2_matched_harm_pareto.png")
    report = ["# I3-2 匹配损害预算的收益—风险实验", "",
              "所有主要控制器共享冻结候选mixer；强度由selection在0.5%、1%、2%和5%预算内选择。", "",
              "## 预注册判定", "", markdown_table(decision), "",
              "## 2%预算主要比较", "", markdown_table(tests), "",
              "## 2%测试损害匹配的描述性Pareto包络", "", markdown_table(matched), "",
              "## 匹配预算摘要", "", markdown_table(summary)]
    (RESULTS/"i3_2_report.md").write_text("\n".join(report), encoding="utf-8")
    print(decision.to_string(index=False))
    print(primary[["dataset", "test_nll_gain_mean", "test_accuracy_gain_mean",
                   "test_clipped_harm_mean", "test_harm_p95_mean",
                   "test_negative_flip_rate_mean", "test_correction_rate_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
