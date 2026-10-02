"""Summarize E10 missing-modality and arbitrary-coalition experiment."""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"MOSI": "mosi", "MOSEI": "mosei", "CREMA-D": "cremad", "AV-MNIST": "avmnist"}
METHODS = ["full_zeroing", "modality_dropout", "coalition_dropout", "availability_safe_rcg"]
LABELS = {"full_zeroing": "full-only + zeroing", "modality_dropout": "modality dropout",
          "coalition_dropout": "coalition dropout", "availability_safe_rcg": "availability-safe RCG"}
COLORS = {"full_zeroing": "#999999", "modality_dropout": "#E69F00",
          "coalition_dropout": "#0072B2", "availability_safe_rcg": "#009E73"}


def weighted_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows, oracle_rows = [], []
    metric_columns = ["accuracy", "macro_f1", "nll", "brier", "ece", "accuracy_gain", "nll_gain",
                      "negative_flip_rate", "correction_rate", "clipped_harm"]
    oracle_columns = ["reference_nll", "valid_oracle_nll", "reference_regret",
                      "reference_oracle_rate", "oracle_single_rate", "oracle_pair_rate",
                      "oracle_full_available_rate"]
    for dataset, slug in DATASETS.items():
        metrics = pd.read_csv(ROOT / f"runs/formal-e10-{slug}/metrics_by_fold_alliance.csv",
                              dtype={"available_mask": str})
        oracle = pd.read_csv(ROOT / f"runs/formal-e10-{slug}/oracle_by_fold_alliance.csv",
                             dtype={"available_mask": str})
        for (mask, alliance, count, method), frame in metrics.groupby(
                ["available_mask", "available_modalities", "available_count", "method"]):
            weights = frame.n_test.to_numpy(float)
            row = {"dataset": dataset, "available_mask": str(mask), "available_modalities": alliance,
                   "available_count": int(count), "method": method, "n_test": int(weights.sum())}
            row.update({column: float(np.average(frame[column], weights=weights))
                        for column in metric_columns})
            for optional in ("mean_alpha", "mean_strength"):
                valid = frame[optional].dropna() if optional in frame else []
                row[optional] = float(np.average(frame.loc[valid.index, optional],
                                                  weights=frame.loc[valid.index, "n_test"])) if len(valid) else np.nan
            metric_rows.append(row)
        for (mask, alliance, count), frame in oracle.groupby(
                ["available_mask", "available_modalities", "available_count"]):
            weights = frame.n_test.to_numpy(float)
            row = {"dataset": dataset, "available_mask": str(mask), "available_modalities": alliance,
                   "available_count": int(count), "n_test": int(weights.sum())}
            row.update({column: float(np.average(frame[column], weights=weights))
                        for column in oracle_columns})
            oracle_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(oracle_rows)


def prediction_differences(dataset: str, slug: str, candidate: str, reference: str,
                           scope: str = "incomplete") -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / f"runs/formal-e10-{slug}/predictions.parquet")
    frame = frame[frame.method.isin([candidate, reference])].copy()
    maximum = frame.available_mask.astype(str).str.len().max()
    if scope == "incomplete":
        frame = frame[frame.available_mask.astype(str).str.count("1") < maximum]
    elif scope == "full":
        frame = frame[frame.available_mask.astype(str).str.count("1") == maximum]
    elif scope != "all":
        raise ValueError(scope)
    pcols = sorted([column for column in frame if re.fullmatch(r"p\d+", column)],
                   key=lambda value: int(value[1:]))
    p = frame[pcols].to_numpy(float); y = frame.label.to_numpy(int)
    frame["loss"] = -np.log(np.clip(p[np.arange(len(frame)), y], 1e-12, 1))
    frame["correct"] = (p.argmax(1) == y).astype(float)
    keys = ["fold", "sample_id", "group_or_video_id", "label", "method"]
    average = frame.groupby(keys, as_index=False)[["loss", "correct"]].mean()
    identity = ["fold", "sample_id", "group_or_video_id", "label"]
    wide = average.pivot(index=identity, columns="method", values=["loss", "correct"]).reset_index()
    return pd.DataFrame({
        "cluster": wide["group_or_video_id"].astype(str), "label": wide["label"].astype(int),
        "nll_difference": wide[("loss", candidate)]-wide[("loss", reference)],
        "accuracy_difference": wide[("correct", candidate)]-wide[("correct", reference)],
    })


def bootstrap(dataset: str, slug: str, candidate: str, reference: str,
              repetitions: int = 10_000, scope: str = "incomplete") -> dict:
    values = prediction_differences(dataset, slug, candidate, reference, scope)
    rng = np.random.default_rng(102026); nll = np.empty(repetitions); accuracy = np.empty(repetitions)
    if dataset == "AV-MNIST":
        groups = [np.flatnonzero(values.label.to_numpy() == label) for label in sorted(values.label.unique())]
        for iteration in range(repetitions):
            selected = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
            nll[iteration] = values.nll_difference.to_numpy()[selected].mean()
            accuracy[iteration] = values.accuracy_difference.to_numpy()[selected].mean()
        unit = "class-stratified sample"
    else:
        sums = values.groupby("cluster")[["nll_difference", "accuracy_difference"]].sum()
        counts = values.groupby("cluster").size().reindex(sums.index).to_numpy()
        for iteration in range(repetitions):
            selected = rng.integers(0, len(sums), len(sums)); denominator = counts[selected].sum()
            nll[iteration] = sums.nll_difference.to_numpy()[selected].sum()/denominator
            accuracy[iteration] = sums.accuracy_difference.to_numpy()[selected].sum()/denominator
        unit = "actor" if dataset == "CREMA-D" else "original video"
    return {"dataset": dataset, "candidate": candidate, "reference": reference,
            "scope": scope, "bootstrap_unit": unit,
            "n_original_samples": len(values), "nll_difference": values.nll_difference.mean(),
            "nll_ci_low": np.quantile(nll, .025), "nll_ci_high": np.quantile(nll, .975),
            "accuracy_difference": values.accuracy_difference.mean(),
            "accuracy_ci_low": np.quantile(accuracy, .025),
            "accuracy_ci_high": np.quantile(accuracy, .975)}


def incomplete_summary(metrics: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        maximum = metrics.loc[metrics.dataset == dataset, "available_count"].max()
        part = metrics[(metrics.dataset == dataset) & (metrics.available_count < maximum)]
        row = {"dataset": dataset, "n_incomplete_alliances": part.available_mask.nunique()}
        for method in METHODS:
            values = part[part.method == method]
            row[f"{method}_accuracy"] = values.accuracy.mean()
            row[f"{method}_nll"] = values.nll.mean()
        oracle_part = oracle[(oracle.dataset == dataset) & (oracle.available_count < maximum)]
        row["reference_regret"] = oracle_part.reference_regret.mean()
        row["reference_oracle_rate"] = oracle_part.reference_oracle_rate.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def plot(metrics: pd.DataFrame) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.2, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.4), constrained_layout=True)
    for axis, dataset in zip(axes.flat, DATASETS):
        part = metrics[metrics.dataset == dataset]
        alliance_order = (part[["available_mask", "available_modalities", "available_count"]]
                          .drop_duplicates().sort_values(["available_count", "available_mask"]))
        labels = alliance_order.available_modalities.tolist(); x = np.arange(len(labels))
        for method in METHODS:
            values = part[part.method == method].set_index("available_mask").loc[
                alliance_order.available_mask.astype(str), "nll"]
            axis.plot(x, values, marker="o", markersize=3, linewidth=1.2,
                      color=COLORS[method], label=LABELS[method])
        axis.set_title(dataset, loc="left", fontweight="bold")
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylabel("NLL")
        axis.grid(axis="y", linewidth=.35, alpha=.35)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6.8)
    fig.savefig(ROOT / "figures/e10_missing_coalitions.png", dpi=300,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics, oracle = weighted_summary()
    metrics.to_csv(ROOT / "results/e10_alliance_metrics.csv", index=False)
    oracle.to_csv(ROOT / "results/e10_alliance_oracle.csv", index=False)
    summary = incomplete_summary(metrics, oracle)
    summary.to_csv(ROOT / "results/e10_incomplete_summary.csv", index=False)
    rows = []
    for dataset, slug in DATASETS.items():
        for candidate, reference in [("coalition_dropout", "full_zeroing"),
                                     ("coalition_dropout", "modality_dropout"),
                                     ("availability_safe_rcg", "coalition_dropout")]:
            rows.append(bootstrap(dataset, slug, candidate, reference))
        rows.append(bootstrap(dataset, slug, "availability_safe_rcg",
                              "coalition_dropout", scope="full"))
    bootstrap_frame = pd.DataFrame(rows)
    bootstrap_frame.to_csv(ROOT / "results/e10_missing_bootstrap.csv", index=False)
    plot(metrics)
    print("\nIncomplete-alliance summary:")
    print(summary.to_string(index=False))
    print("\nPaired bootstrap:")
    print(bootstrap_frame.to_string(index=False))


if __name__ == "__main__":
    main()
