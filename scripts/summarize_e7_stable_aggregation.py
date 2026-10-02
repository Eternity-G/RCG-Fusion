"""Summarize E7 with paired replicate tests and cluster bootstrap."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"MOSI": "mosi", "MOSEI": "mosei", "CREMA-D": "cremad", "AV-MNIST": "avmnist"}
METHODS = ["full_ensemble", "posterior_ensemble", "equal_full_posterior",
           "logit_stacking", "simplex_convex", "simplex_safe_fallback"]
LABELS = {"full_ensemble": "Full ensemble", "posterior_ensemble": "Posterior ensemble",
          "equal_full_posterior": "Equal full + posterior", "logit_stacking": "Logit stacking",
          "simplex_convex": "Simplex convex", "simplex_safe_fallback": "Convex + fallback"}
COLORS = {"full_ensemble": "#777777", "posterior_ensemble": "#E69F00",
          "equal_full_posterior": "#56B4E9", "logit_stacking": "#CC79A7",
          "simplex_convex": "#009E73", "simplex_safe_fallback": "#0072B2"}


def ci95(values):
    values = np.asarray(values, float)
    if len(values) < 2: return np.nan
    return float(stats.t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))


def holm(values):
    values = np.asarray(values, float); order = np.argsort(values)
    result = np.empty(len(values)); running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(values)-rank)*values[index]))
        result[index] = running
    return result


def metric_summary(groups):
    rows = []
    for (dataset, method), values in groups.groupby(["dataset", "method"]):
        row = {"dataset": dataset, "method": method, "n_groups": len(values)}
        for metric in ["accuracy", "macro_f1", "nll", "brier", "ece", "accuracy_gain",
                       "nll_gain", "relative_error_reduction", "relative_nll_reduction",
                       "negative_flip_rate", "correction_rate", "clipped_harm"]:
            row[f"{metric}_mean"] = values[metric].mean()
            row[f"{metric}_sd"] = values[metric].std(ddof=1) if len(values) > 1 else np.nan
            row[f"{metric}_ci95"] = ci95(values[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(groups):
    rows = []
    for dataset in ("MOSI", "MOSEI"):
        part = groups[groups.dataset == dataset]
        baseline = part[part.method == "full_ensemble"].sort_values("group")
        for method in METHODS[1:]:
            candidate = part[part.method == method].sort_values("group")
            for metric in ("accuracy", "nll", "negative_flip_rate", "clipped_harm"):
                difference = candidate[metric].to_numpy()-baseline[metric].to_numpy()
                test = stats.ttest_1samp(difference, 0)
                rows.append({"dataset": dataset, "method": method, "metric": metric,
                             "mean_difference": difference.mean(), "ci95": ci95(difference),
                             "t_statistic": test.statistic, "p_value": test.pvalue})
    adjusted = holm([row["p_value"] for row in rows])
    for row, value in zip(rows, adjusted): row["holm_p"] = value
    return pd.DataFrame(rows)


def paired_bootstrap(frame, dataset, repetitions=10_000, seed=7717):
    """Bootstrap Convex+fallback minus full, averaging independent ensembles."""
    subset = frame[(frame.dataset == dataset) &
                   (frame.method.isin(["full_ensemble", "simplex_safe_fallback"]))].copy()
    keys = ["group", "fold", "sample_id", "group_or_video_id", "label"]
    pcols = sorted([c for c in subset if c.startswith("p") and not subset[c].isna().all()],
                   key=lambda x: int(x[1:]))
    value = subset.pivot(index=keys, columns="method", values=pcols)
    # Average paired per-sample deltas over independent ensemble groups.
    records = []
    for index, row in value.iterrows():
        group, fold, sample, cluster, label = index
        full = np.array([row[(c, "full_ensemble")] for c in pcols])
        safe = np.array([row[(c, "simplex_safe_fallback")] for c in pcols])
        records.append({"group": group, "fold": fold, "sample_id": sample, "cluster": cluster,
                        "label": label, "acc_delta": float(safe.argmax() == label)-float(full.argmax() == label),
                        "nll_delta": -np.log(np.clip(safe[label], 1e-12, 1)) + np.log(np.clip(full[label], 1e-12, 1))})
    values = pd.DataFrame(records)
    values = values.groupby(["fold", "sample_id", "cluster", "label"], as_index=False)[["acc_delta", "nll_delta"]].mean()
    rng = np.random.default_rng(seed); acc = np.empty(repetitions); nll = np.empty(repetitions)
    if dataset == "AV-MNIST":
        class_indices = [np.flatnonzero(values.label.to_numpy() == label) for label in sorted(values.label.unique())]
        for draw in range(repetitions):
            index = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in class_indices])
            acc[draw] = values.acc_delta.to_numpy()[index].mean()
            nll[draw] = values.nll_delta.to_numpy()[index].mean()
        cluster_type = "class-stratified sample"
    else:
        cluster = values.cluster.astype(str).to_numpy()
        unique = np.unique(cluster)
        sums = values.assign(cluster=cluster).groupby("cluster")[["acc_delta", "nll_delta"]].sum().reindex(unique)
        counts = values.assign(cluster=cluster).groupby("cluster").size().reindex(unique).to_numpy()
        for draw in range(repetitions):
            index = rng.integers(0, len(unique), size=len(unique))
            denominator = counts[index].sum()
            acc[draw] = sums.acc_delta.to_numpy()[index].sum()/denominator
            nll[draw] = sums.nll_delta.to_numpy()[index].sum()/denominator
        cluster_type = "actor" if dataset == "CREMA-D" else "original video"
    return {"dataset": dataset, "method": "simplex_safe_fallback", "comparison": "vs_full_ensemble",
            "bootstrap_repetitions": repetitions, "cluster": cluster_type,
            "accuracy_delta": values.acc_delta.mean(), "accuracy_ci_low": np.quantile(acc, .025),
            "accuracy_ci_high": np.quantile(acc, .975), "nll_delta": values.nll_delta.mean(),
            "nll_ci_low": np.quantile(nll, .025), "nll_ci_high": np.quantile(nll, .975)}


def plot(summary):
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for panel, (dataset, axis) in zip("abcd", zip(DATASETS, axes.flat)):
        part = summary[summary.dataset == dataset].set_index("method").reindex(METHODS)
        x = np.arange(len(METHODS)); gain = part.nll_gain_mean.to_numpy()
        err = np.nan_to_num(part.nll_gain_ci95.to_numpy(), nan=0.)
        axis.bar(x, gain, yerr=err, capsize=2, color=[COLORS[m] for m in METHODS], width=.72)
        axis.axhline(0, color="#333333", linewidth=.7)
        if dataset == "CREMA-D":
            # Keep the informative small differences readable while explicitly
            # marking the severe off-scale failure of unconstrained stacking.
            axis.set_ylim(-.022, .018)
            stacking_index = METHODS.index("logit_stacking")
            axis.scatter(stacking_index, -.0205, marker="v", s=22,
                         color=COLORS["logit_stacking"], clip_on=False, zorder=4)
            axis.text(stacking_index, -.0188, f"off-scale\n{gain[stacking_index]:.3f}",
                      ha="center", va="bottom", fontsize=6.2, fontweight="bold",
                      color="white")
        axis.set_xticks(x, [LABELS[m] for m in METHODS], rotation=32, ha="right")
        axis.set_ylabel("NLL improvement vs full ensemble")
        axis.set_title(f"({panel}) {dataset}", loc="left", fontweight="bold")
        axis.grid(axis="y", color="#e5e5e5", linewidth=.6, zorder=0)
    fig.savefig(ROOT/"figures/e7_stable_aggregation.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    groups, predictions, weights, fallbacks = [], [], [], []
    for dataset, slug in DATASETS.items():
        run = ROOT/f"runs/formal-e7-{slug}"
        group = pd.read_csv(run/"metrics_by_group.csv"); group["dataset"] = dataset; groups.append(group)
        pred = pd.read_parquet(run/"predictions.parquet"); pred["dataset"] = dataset; predictions.append(pred)
        weight = pd.read_csv(run/"simplex_weights.csv"); weight["dataset"] = dataset; weights.append(weight)
        fallback = pd.read_csv(run/"safe_fallback.csv"); fallback["dataset"] = dataset; fallbacks.append(fallback)
    groups = pd.concat(groups, ignore_index=True); predictions = pd.concat(predictions, ignore_index=True)
    weights = pd.concat(weights, ignore_index=True); fallbacks = pd.concat(fallbacks, ignore_index=True)
    result = ROOT/"results"; result.mkdir(exist_ok=True)
    groups.to_csv(result/"e7_aggregation_by_group.csv", index=False)
    summary = metric_summary(groups); summary.to_csv(result/"e7_aggregation_summary.csv", index=False)
    paired_tests(groups).to_csv(result/"e7_aggregation_paired_tests.csv", index=False)
    pd.DataFrame([paired_bootstrap(predictions, dataset) for dataset in DATASETS]).to_csv(
        result/"e7_aggregation_bootstrap.csv", index=False)
    weights.to_csv(result/"e7_simplex_weights.csv", index=False)
    fallbacks.to_csv(result/"e7_safe_fallback.csv", index=False)
    plot(summary)
    columns = ["dataset", "method", "accuracy_mean", "accuracy_gain_mean", "nll_mean", "nll_gain_mean",
               "negative_flip_rate_mean", "correction_rate_mean", "clipped_harm_mean"]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__": main()
