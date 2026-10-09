"""Analyze I3-3 model-level aggregation and stable fallback variants."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT/"results", ROOT/"figures"
DATASETS = ("mosi", "mosei", "cremad", "avmnist")
METHODS = ("full_ensemble", "temperature_full_ensemble", "a7_equal_ensemble",
           "posterior_ensemble", "all_actions_equal", "logit_stacking",
           "simplex_safe_fallback", "hierarchical_safe_fallback",
           "a7_member_safe_fallback", "regularized_a7_safe_fallback")
LABELS = {"full_ensemble": "Full ensemble", "temperature_full_ensemble": "Temp. full",
          "a7_equal_ensemble": "A7 average", "posterior_ensemble": "Posterior average",
          "all_actions_equal": "Equal full+A7", "logit_stacking": "Logit stacking",
          "simplex_safe_fallback": "Joint simplex + fallback",
          "hierarchical_safe_fallback": "Hierarchical 2-action",
          "a7_member_safe_fallback": "A7 simplex + fallback",
          "regularized_a7_safe_fallback": "Stable regularized A7 + fallback"}
COLORS = {"full_ensemble": "#7F7F7F", "temperature_full_ensemble": "#BDBDBD",
          "a7_equal_ensemble": "#56B4E9", "posterior_ensemble": "#E69F00",
          "all_actions_equal": "#F0E442", "logit_stacking": "#CC79A7",
          "simplex_safe_fallback": "#0072B2", "hierarchical_safe_fallback": "#009E73",
          "a7_member_safe_fallback": "#E69F00",
          "regularized_a7_safe_fallback": "#D55E00"}


def load():
    metrics, predictions, weights, fallbacks = [], [], [], []
    for dataset in DATASETS:
        root = ROOT/"runs"/f"formal-i3-3-{dataset}"
        metric = pd.read_csv(root/"metrics.csv"); metric["dataset"] = dataset; metrics.append(metric)
        prediction = pd.read_parquet(root/"predictions.parquet"); predictions.append(prediction)
        weight = pd.read_csv(root/"weights.csv"); weights.append(weight)
        fallback = pd.read_csv(root/"fallbacks.csv"); fallbacks.append(fallback)
    return (pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True),
            pd.concat(weights, ignore_index=True), pd.concat(fallbacks, ignore_index=True))


def paired_cluster_bootstrap(frame, left, right, repetitions=10000, seed=20261007):
    pivot = frame[frame.method.isin((left, right))].copy()
    pcols = sorted([column for column in pivot if column.startswith("p") and
                    pivot[column].notna().any()],
                   key=lambda value: int(value[1:]))
    records = {}
    for method, values in pivot.groupby("method"):
        values = values.sort_values(["fold", "sample_id"])
        labels = values.label.to_numpy(int); row = np.arange(len(values))
        probability = values[pcols].to_numpy()
        records[method] = {"loss": -np.log(np.clip(probability[row, labels], 1e-12, 1)),
                           "correct": (probability.argmax(1) == labels).astype(float),
                           "group": values.group_or_video_id.astype(str).to_numpy(),
                           "label": labels,
                           "key": list(zip(values.fold, values.sample_id.astype(str)))}
    if records[left]["key"] != records[right]["key"]:
        raise AssertionError("prediction rows are not paired")
    loss_difference = records[right]["loss"]-records[left]["loss"]
    accuracy_difference = records[left]["correct"]-records[right]["correct"]
    groups = records[left]["group"]; rng = np.random.default_rng(seed)
    if frame.dataset.iloc[0] == "avmnist":
        strata = [np.flatnonzero(records[left]["label"] == label)
                  for label in np.unique(records[left]["label"])]
        estimates = []
        for _ in range(repetitions):
            index = np.concatenate([rng.choice(part, len(part), replace=True) for part in strata])
            estimates.append((loss_difference[index].mean(), accuracy_difference[index].mean()))
    else:
        unique = np.unique(groups)
        sums_loss = np.array([loss_difference[groups == group].sum() for group in unique])
        sums_accuracy = np.array([accuracy_difference[groups == group].sum() for group in unique])
        counts = np.array([(groups == group).sum() for group in unique])
        estimates = []
        for _ in range(repetitions):
            index = rng.integers(0, len(unique), len(unique))
            estimates.append((sums_loss[index].sum()/counts[index].sum(),
                              sums_accuracy[index].sum()/counts[index].sum()))
    estimates = np.asarray(estimates)
    return {"nll_difference": float(loss_difference.mean()),
            "nll_ci_low": float(np.quantile(estimates[:, 0], .025)),
            "nll_ci_high": float(np.quantile(estimates[:, 0], .975)),
            "accuracy_difference": float(accuracy_difference.mean()),
            "accuracy_ci_low": float(np.quantile(estimates[:, 1], .025)),
            "accuracy_ci_high": float(np.quantile(estimates[:, 1], .975))}


def bootstrap_all(predictions):
    rows = []
    comparisons = (("regularized_a7_safe_fallback", "full_ensemble", "stable_vs_full"),
                   ("regularized_a7_safe_fallback", "a7_equal_ensemble", "stable_vs_a7"),
                   ("simplex_safe_fallback", "a7_equal_ensemble", "joint_vs_a7"))
    for dataset in DATASETS:
        part = predictions[predictions.dataset == dataset]
        for left, right, name in comparisons:
            rows.append({"dataset": dataset, "comparison": name,
                         **paired_cluster_bootstrap(part, left, right)})
    return pd.DataFrame(rows)


def plot(metrics, fallbacks, path):
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.2, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.8), constrained_layout=True)
    names = ("MOSI", "MOSEI", "CREMA-D", "AV-MNIST"); x = np.arange(4)
    shown = ("a7_equal_ensemble", "simplex_safe_fallback",
             "hierarchical_safe_fallback", "regularized_a7_safe_fallback")
    width = .18
    for axis, metric, title, scale in ((axes[0, 0], "nll_gain", "a  NLL gain over full ensemble", 1),
                                       (axes[0, 1], "accuracy_gain", "b  Accuracy gain", 100)):
        for index, method in enumerate(shown):
            values = metrics[metrics.method == method].set_index("dataset").loc[list(DATASETS)]
            axis.bar(x+(index-1.5)*width, values[metric]*scale, width,
                     color=COLORS[method], label=LABELS[method], zorder=3)
        axis.axhline(0, color="#666666", lw=.7); axis.set_xticks(x, names, rotation=15, ha="right")
        axis.set_ylabel("NLL reduction" if metric == "nll_gain" else "Percentage points")
        axis.set_title(title, loc="left", fontweight="bold"); axis.grid(axis="y", color="#E2E2E2", lw=.55)
    axes[0, 0].legend(ncol=2, fontsize=6.4)

    harm_methods = ("logit_stacking", "simplex_safe_fallback",
                    "regularized_a7_safe_fallback")
    hwidth = .23
    for index, method in enumerate(harm_methods):
        values = metrics[metrics.method == method].set_index("dataset").loc[list(DATASETS)]
        axes[1, 0].bar(x+(index-1)*hwidth, 100*values.clipped_harm, hwidth,
                       color=COLORS[method], label=LABELS[method], zorder=3)
    axes[1, 0].set_xticks(x, names, rotation=15, ha="right"); axes[1, 0].set_ylabel("Mean clipped harm (%)")
    axes[1, 0].set_title("c  Tail-risk control", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="y", color="#E2E2E2", lw=.55); axes[1, 0].legend(fontsize=6.3)

    weight_rows = []
    for dataset in DATASETS:
        path_root = ROOT/"runs"/f"formal-i3-3-{dataset}"
        weight = pd.read_csv(path_root/"weights.csv")
        a7 = weight[weight.action.str.startswith("regularized_a7_member_")].groupby("fold").apply(
            lambda values: float((values.weight > 1e-4).sum()), include_groups=False).mean()
        fallback = pd.read_csv(path_root/"fallbacks.csv")
        weight_rows.append((a7, fallback.regularized_a7_rho.mean(),
                            fallback.rho.mean()))
    weight_rows = np.asarray(weight_rows)
    axes[1, 1].bar(x-.2, weight_rows[:, 0], .4, color="#D55E00", label="Active A7 members")
    axes[1, 1].set_ylabel("Active members")
    secondary = axes[1, 1].twinx()
    secondary.plot(x, weight_rows[:, 1], color="#0072B2", marker="o", label="Stable fallback rho")
    secondary.plot(x, weight_rows[:, 2], color="#7F7F7F", marker="s", label="Joint fallback rho")
    secondary.set_ylabel("Mean fallback coefficient")
    axes[1, 1].set_xticks(x, names, rotation=15, ha="right")
    axes[1, 1].set_title("d  Sparsity and fallback", loc="left", fontweight="bold")
    handles1, labels1 = axes[1, 1].get_legend_handles_labels()
    handles2, labels2 = secondary.get_legend_handles_labels()
    axes[1, 1].legend(handles1+handles2, labels1+labels2, fontsize=6.3, loc="upper right")
    fig.savefig(path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame):
    def fmt(value): return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "|" + "|".join("---" for _ in frame.columns) + "|",
                      *("| " + " | ".join(fmt(value) for value in row) + " |"
                        for row in frame.itertuples(index=False, name=None))])


def main():
    metrics, predictions, weights, fallbacks = load()
    metrics = metrics[metrics.method.isin(METHODS)]
    bootstrap = bootstrap_all(predictions)
    metrics.to_csv(RESULTS/"i3_3_aggregation_summary.csv", index=False)
    bootstrap.to_csv(RESULTS/"i3_3_bootstrap.csv", index=False)
    weights.to_csv(RESULTS/"i3_3_weights.csv", index=False)
    fallbacks.to_csv(RESULTS/"i3_3_fallbacks.csv", index=False)
    stable = metrics[metrics.method == "regularized_a7_safe_fallback"].set_index("dataset")
    a7 = metrics[metrics.method == "a7_equal_ensemble"].set_index("dataset")
    logit = metrics[metrics.method == "logit_stacking"].set_index("dataset")
    decision = pd.DataFrame([
        {"criterion": "stable aggregation improves NLL over A7 average",
         "datasets_passed": int((stable.nll < a7.nll).sum()), "required": 3},
        {"criterion": "stable aggregation improves NLL over full ensemble",
         "datasets_passed": int((stable.nll_gain > 0).sum()), "required": 3},
        {"criterion": "accuracy loss is no worse than 0.5pp",
         "datasets_passed": int((stable.accuracy_gain >= -.005).sum()), "required": 4},
        {"criterion": "nonnegative stable aggregation lowers logit-stacking harm",
         "datasets_passed": int((stable.clipped_harm < logit.clipped_harm).sum()), "required": 3},
    ])
    decision["pass"] = decision.datasets_passed >= decision.required
    decision.to_csv(RESULTS/"i3_3_decision.csv", index=False)
    plot(metrics, fallbacks, FIGURES/"i3_3_stable_aggregation.png")
    report = ["# I3-3 模型级稳定聚合与完整联盟回退", "", "## 预注册判定", "",
              markdown_table(decision), "", "## 方法摘要", "", markdown_table(metrics), "",
              "## 配对簇Bootstrap", "", markdown_table(bootstrap)]
    (RESULTS/"i3_3_report.md").write_text("\n".join(report), encoding="utf-8")
    print(decision.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
