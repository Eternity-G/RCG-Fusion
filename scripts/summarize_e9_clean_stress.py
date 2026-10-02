"""Summarize E9 clean-trained feature degradation experiments.

The stress protocol is evaluated with frozen clean checkpoints. Repeated
corruptions are first averaged within each original sample before bootstrap,
so corruption seeds and modalities are not treated as independent samples.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"MOSI": "mosi", "MOSEI": "mosei", "CREMA-D": "cremad", "AV-MNIST": "avmnist"}
METHODS = [
    "full_ensemble",
    "posterior_ensemble",
    "raw_mixer_ensemble",
    "confidence_shrink_ensemble",
    "entropy_shrink_ensemble",
    "A7_equal_ensemble",
    "A8_safe_fallback",
]
PLOT_METHODS = [
    "confidence_shrink_ensemble",
    "entropy_shrink_ensemble",
    "A7_equal_ensemble",
    "A8_safe_fallback",
]
LABELS = {
    "confidence_shrink_ensemble": "confidence shrink",
    "entropy_shrink_ensemble": "entropy shrink",
    "A7_equal_ensemble": "A7 adaptive shrink",
    "A8_safe_fallback": "A8 safe fallback",
}
COLORS = {
    "confidence_shrink_ensemble": "#E69F00",
    "entropy_shrink_ensemble": "#CC79A7",
    "A7_equal_ensemble": "#0072B2",
    "A8_safe_fallback": "#009E73",
}


def load_metrics() -> pd.DataFrame:
    frames = []
    for dataset, slug in DATASETS.items():
        frame = pd.read_csv(ROOT / f"runs/formal-e9-clean-{slug}/metrics_by_condition_fold.csv")
        frame["dataset_label"] = dataset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def weighted_mean(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    weights = frame["n_test"].to_numpy(float)
    return {column: float(np.average(frame[column], weights=weights)) for column in columns}


def make_curves(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average folds and corruption seeds, retaining type/modality/level."""
    numeric = [
        "accuracy", "macro_f1", "nll", "brier", "ece", "accuracy_gain", "nll_gain",
        "negative_flip_rate", "correction_rate", "clipped_harm", "mean_a7_alpha",
        "mean_full_weight", "mean_posterior_weight", "contribution_auroc",
    ]
    rows = []
    keys = ["dataset_label", "method", "corruption_type", "affected_modality", "corruption_level"]
    for group, frame in metrics.groupby(keys, dropna=False):
        row = dict(zip(keys, group))
        row["n_test_fold_sum"] = int(frame.groupby("fold")["n_test"].first().sum())
        row["n_corruption_seeds"] = int(frame["corruption_seed"].nunique())
        row.update(weighted_mean(frame, numeric))
        rows.append(row)
    return pd.DataFrame(rows)


def clean_rows(curves: pd.DataFrame, dataset: str) -> pd.DataFrame:
    return curves[(curves.dataset_label == dataset) & (curves.corruption_type == "clean")]


def severity_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Average affected modalities at each severity and add clean severity zero."""
    rows = []
    numeric = ["accuracy", "macro_f1", "nll", "brier", "ece", "negative_flip_rate",
               "correction_rate", "clipped_harm", "mean_a7_alpha", "contribution_auroc"]
    for dataset in DATASETS:
        clean = clean_rows(curves, dataset).set_index("method")
        for corruption in ["gaussian", "mask"]:
            stressed = curves[(curves.dataset_label == dataset) & (curves.corruption_type == corruption)]
            for method in METHODS:
                if method not in clean.index:
                    continue
                base = clean.loc[method]
                row = {"dataset": dataset, "method": method, "corruption_type": corruption,
                       "severity_index": 0, "corruption_level": 0.0, "n_modalities": 0}
                row.update({column: float(base[column]) for column in numeric})
                rows.append(row)
                levels = sorted(stressed.corruption_level.unique())
                for index, level in enumerate(levels, start=1):
                    part = stressed[(stressed.method == method) & (stressed.corruption_level == level)]
                    row = {"dataset": dataset, "method": method, "corruption_type": corruption,
                           "severity_index": index, "corruption_level": float(level),
                           "n_modalities": int(part.affected_modality.nunique())}
                    row.update({column: float(part[column].mean()) for column in numeric})
                    rows.append(row)
    output = pd.DataFrame(rows)
    reference = output[output.method == "full_ensemble"][["dataset", "corruption_type", "severity_index", "accuracy", "nll"]]
    reference = reference.rename(columns={"accuracy": "full_accuracy", "nll": "full_nll"})
    output = output.merge(reference, on=["dataset", "corruption_type", "severity_index"], how="left")
    output["accuracy_gain_vs_full"] = output.accuracy - output.full_accuracy
    output["nll_gain_vs_full"] = output.full_nll - output.nll
    return output


def auc_summary(severity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, corruption), frame in severity.groupby(["dataset", "method", "corruption_type"]):
        frame = frame.sort_values("severity_index")
        max_index = float(frame.severity_index.max())
        x = frame.severity_index.to_numpy(float) / max_index
        rows.append({
            "dataset": dataset,
            "method": method,
            "corruption_type": corruption,
            "accuracy_auc": float(np.trapz(frame.accuracy, x)),
            "macro_f1_auc": float(np.trapz(frame.macro_f1, x)),
            "nll_auc": float(np.trapz(frame.nll, x)),
            "mean_nll_gain_vs_full": float(frame.nll_gain_vs_full.iloc[1:].mean()),
            "mean_accuracy_gain_vs_full": float(frame.accuracy_gain_vs_full.iloc[1:].mean()),
            "worst_accuracy": float(frame.accuracy.min()),
            "worst_nll": float(frame.nll.max()),
        })
    return pd.DataFrame(rows)


def worst_modality(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stressed = curves[curves.corruption_type.isin(["gaussian", "mask"])]
    for (dataset, method, corruption), frame in stressed.groupby(["dataset_label", "method", "corruption_type"]):
        maximum = frame.corruption_level.max()
        severe = frame[frame.corruption_level == maximum]
        for _, row in severe.iterrows():
            rows.append({"dataset": dataset, "method": method, "corruption_type": corruption,
                         "level": maximum, "affected_modality": row.affected_modality,
                         "accuracy": row.accuracy, "macro_f1": row.macro_f1, "nll": row.nll,
                         "nll_gain_vs_full": row.nll_gain})
    return pd.DataFrame(rows)


def load_prediction_means(dataset: str, slug: str, method: str) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / f"runs/formal-e9-clean-{slug}/predictions.parquet")
    frame = frame[(frame.method == method) & (frame.corruption_type != "clean")].copy()
    pcols = sorted([column for column in frame if column.startswith("p")], key=lambda x: int(x[1:]))
    probability = frame[pcols].to_numpy(float)
    labels = frame.label.to_numpy(int)
    frame["sample_nll"] = -np.log(np.clip(probability[np.arange(len(frame)), labels], 1e-12, 1.0))
    frame["sample_correct"] = (probability.argmax(axis=1) == labels).astype(float)
    keys = ["fold", "sample_id", "group_or_video_id", "label"]
    return frame.groupby(keys, as_index=False)[["sample_nll", "sample_correct"]].mean()


def paired_bootstrap(dataset: str, slug: str, candidate: str, repetitions: int = 10_000) -> dict:
    base = load_prediction_means(dataset, slug, "full_ensemble")
    candidate_frame = load_prediction_means(dataset, slug, candidate)
    keys = ["fold", "sample_id", "group_or_video_id", "label"]
    merged = base.merge(candidate_frame, on=keys, suffixes=("_base", "_candidate"), validate="one_to_one")
    merged["nll_difference"] = merged.sample_nll_candidate - merged.sample_nll_base
    merged["accuracy_difference"] = merged.sample_correct_candidate - merged.sample_correct_base
    rng = np.random.default_rng(92026)
    nll = np.empty(repetitions)
    accuracy = np.empty(repetitions)
    if dataset == "AV-MNIST":
        groups = [np.flatnonzero(merged.label.to_numpy() == label) for label in sorted(merged.label.unique())]
        for index in range(repetitions):
            sampled = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
            nll[index] = merged.nll_difference.to_numpy()[sampled].mean()
            accuracy[index] = merged.accuracy_difference.to_numpy()[sampled].mean()
        unit = "class-stratified sample"
    else:
        cluster = merged.group_or_video_id.astype(str)
        sums = merged.assign(cluster=cluster).groupby("cluster")[["nll_difference", "accuracy_difference"]].sum()
        counts = merged.assign(cluster=cluster).groupby("cluster").size().reindex(sums.index).to_numpy()
        for index in range(repetitions):
            sampled = rng.integers(0, len(sums), len(sums))
            denominator = counts[sampled].sum()
            nll[index] = sums.nll_difference.to_numpy()[sampled].sum() / denominator
            accuracy[index] = sums.accuracy_difference.to_numpy()[sampled].sum() / denominator
        unit = "actor" if dataset == "CREMA-D" else "original video"
    return {
        "dataset": dataset, "candidate": candidate, "reference": "full_ensemble", "bootstrap_unit": unit,
        "n_original_samples": len(merged), "nll_difference": merged.nll_difference.mean(),
        "nll_ci_low": np.quantile(nll, .025), "nll_ci_high": np.quantile(nll, .975),
        "accuracy_difference": merged.accuracy_difference.mean(),
        "accuracy_ci_low": np.quantile(accuracy, .025), "accuracy_ci_high": np.quantile(accuracy, .975),
    }


def plot(severity: pd.DataFrame) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.2,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True, sharey=False)
    for axis, dataset in zip(axes.flat, DATASETS):
        part = severity[severity.dataset == dataset]
        for method in PLOT_METHODS:
            for corruption, style in [("gaussian", "-"), ("mask", "--")]:
                values = part[(part.method == method) & (part.corruption_type == corruption)].sort_values("severity_index")
                axis.plot(values.severity_index, values.nll_gain_vs_full, linestyle=style, marker="o", markersize=2.7,
                          linewidth=1.2, color=COLORS[method], label=f"{LABELS[method]} ({corruption})")
        axis.axhline(0, color="#333333", linewidth=.7)
        axis.set_title(dataset, loc="left", fontweight="bold")
        axis.set_xlabel("Severity index (0 = clean)")
        axis.set_ylabel("NLL gain over full ensemble")
        axis.set_xticks(range(5))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6.2)
    fig.savefig(ROOT / "figures/e9_clean_stress.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    curves = make_curves(metrics)
    severity = severity_curves(curves)
    auc = auc_summary(severity)
    worst = worst_modality(curves)
    bootstraps = []
    for dataset, slug in DATASETS.items():
        for method in ["A7_equal_ensemble", "A8_safe_fallback"]:
            bootstraps.append(paired_bootstrap(dataset, slug, method))
    bootstrap = pd.DataFrame(bootstraps)
    curves.to_csv(ROOT / "results/e9_clean_stress_curves.csv", index=False)
    severity.to_csv(ROOT / "results/e9_clean_stress_severity.csv", index=False)
    auc.to_csv(ROOT / "results/e9_clean_stress_summary.csv", index=False)
    worst.to_csv(ROOT / "results/e9_clean_stress_worst.csv", index=False)
    bootstrap.to_csv(ROOT / "results/e9_clean_stress_bootstrap.csv", index=False)
    plot(severity)
    print("\nStress AUC summary (main methods):")
    print(auc[auc.method.isin(["full_ensemble", "A7_equal_ensemble", "A8_safe_fallback"])].to_string(index=False))
    print("\nPaired bootstrap after averaging all stress repeats per original sample:")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
