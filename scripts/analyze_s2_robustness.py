"""Summarize S2 clean-only/shared-augmentation robustness results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("mosi", "mosei", "cremad", "avmnist")
DISPLAY = {"full_ensemble": "Full", "A7_equal_ensemble": "A7", "A8_safe_fallback": "A8",
           "tmc": "TMC", "qmf": "QMF", "pdf": "PDF", "i2moe": "I²MoE"}
COLORS = {"Full": "#4D4D4D", "A7": "#0072B2", "A8": "#D55E00", "TMC": "#009E73",
          "QMF": "#CC79A7", "PDF": "#E69F00", "I²MoE": "#56B4E9"}


def holm_adjust(values):
    values = np.asarray(values, dtype=float); order = np.argsort(values); adjusted = np.empty_like(values)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(values)-rank)*values[index]); adjusted[index] = min(running, 1.)
    return adjusted


def paired_bootstrap(frame: pd.DataFrame, dataset: str, protocol: str, baseline: str,
                     draws: int = 10_000):
    probability_columns = sorted((c for c in frame if c.startswith("p") and c[1:].isdigit()),
                                 key=lambda value: int(value[1:]))
    keep = frame[(frame.condition != "clean") & frame.method.isin(("A7_equal_ensemble", baseline))].copy()
    probability = keep[probability_columns].to_numpy(float); labels = keep.label.to_numpy(int)
    keep["loss"] = -np.log(np.clip(probability[np.arange(len(labels)), labels], 1e-12, 1))
    keep["correct"] = (probability.argmax(1) == labels).astype(float)
    index = ["fold", "condition", "sample_id"]
    pivot_loss = keep.pivot(index=index, columns="method", values="loss").dropna()
    pivot_acc = keep.pivot(index=index, columns="method", values="correct").dropna()
    metadata = keep.drop_duplicates(index).set_index(index)["group_or_video_id"].astype(str)
    difference = pd.DataFrame({
        "nll_gain": pivot_loss[baseline]-pivot_loss["A7_equal_ensemble"],
        "accuracy_gain": pivot_acc["A7_equal_ensemble"]-pivot_acc[baseline],
        "cluster": [f"{fold}:{metadata.loc[(fold, condition, sample)]}"
                    for fold, condition, sample in pivot_loss.index],
    })
    cluster = difference.groupby("cluster")[["nll_gain", "accuracy_gain"]].mean()
    rng = np.random.default_rng(20261009)
    samples = rng.integers(0, len(cluster), size=(draws, len(cluster)))
    boot = cluster.to_numpy()[samples].mean(1)
    observed = cluster.mean().to_numpy()
    p = [min(1., 2*min((boot[:, j] <= 0).mean(), (boot[:, j] >= 0).mean())) for j in range(2)]
    return {
        "dataset": dataset, "protocol": protocol, "baseline": baseline,
        "n_clusters": len(cluster), "nll_gain": observed[0],
        "nll_ci_low": np.quantile(boot[:, 0], .025), "nll_ci_high": np.quantile(boot[:, 0], .975),
        "nll_p": p[0], "accuracy_gain": observed[1],
        "accuracy_ci_low": np.quantile(boot[:, 1], .025),
        "accuracy_ci_high": np.quantile(boot[:, 1], .975), "accuracy_p": p[1],
    }


def read_predictions(dataset: str, protocol: str):
    if protocol == "clean-only":
        internal = ROOT / f"runs/formal-e9-clean-{dataset}/predictions.parquet"
        strong = ROOT / f"runs/formal-s2-strong-clean/{dataset}/predictions.parquet"
    else:
        internal = ROOT / f"runs/formal-e9-augmented-{dataset}/predictions.parquet"
        strong = ROOT / f"runs/formal-s2-strong-augmented/{dataset}/predictions.parquet"
    frames = []
    if internal.exists():
        frames.append(pd.read_parquet(internal))
    if strong.exists():
        frames.append(pd.read_parquet(strong))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def condition_metrics(frame: pd.DataFrame):
    rows = []
    probability_columns = sorted((c for c in frame if c.startswith("p") and c[1:].isdigit()),
                                 key=lambda value: int(value[1:]))
    keys = ["method", "condition", "corruption_type", "affected_modality",
            "corruption_level", "corruption_seed"]
    for key, part in frame.groupby(keys, dropna=False):
        probability = part[probability_columns].to_numpy(float)
        labels = part.label.to_numpy(int); index = np.arange(len(labels))
        rows.append(dict(zip(keys, key)) | {
            "accuracy": float((probability.argmax(1) == labels).mean()),
            "nll": float(-np.log(np.clip(probability[index, labels], 1e-12, 1)).mean()),
            "n": len(labels),
        })
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame, dataset: str, protocol: str):
    clean = metrics[metrics.condition == "clean"].set_index("method")
    corrupt = metrics[metrics.condition != "clean"]
    rows = []
    for method, part in corrupt.groupby("method"):
        rows.append({
            "dataset": dataset, "protocol": protocol, "method": method,
            "clean_accuracy": clean.loc[method, "accuracy"], "clean_nll": clean.loc[method, "nll"],
            "mean_corrupt_accuracy": part.accuracy.mean(), "mean_corrupt_nll": part.nll.mean(),
            "worst_accuracy": part.accuracy.min(), "worst_nll": part.nll.max(),
            "clean_to_corrupt_accuracy_drop": clean.loc[method, "accuracy"]-part.accuracy.mean(),
            "clean_to_corrupt_nll_increase": part.nll.mean()-clean.loc[method, "nll"],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("clean-only", "shared-augmentation", "all"), default="all")
    args = parser.parse_args()
    protocols = ("clean-only", "shared-augmentation") if args.protocol == "all" else (args.protocol,)
    all_metrics, all_summary, bootstrap_rows = [], [], []
    for protocol in protocols:
        for dataset in DATASETS:
            frame = read_predictions(dataset, protocol)
            if frame is None:
                continue
            metrics = condition_metrics(frame)
            metrics.insert(0, "dataset", dataset); metrics.insert(1, "protocol", protocol)
            all_metrics.append(metrics); all_summary.append(summarize(metrics, dataset, protocol))
            for baseline in ("tmc", "qmf", "pdf", "i2moe"):
                if baseline in set(frame.method):
                    bootstrap_rows.append(paired_bootstrap(frame, dataset, protocol, baseline))
    if not all_metrics:
        raise FileNotFoundError("no S2 prediction files found")
    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    (ROOT / "results").mkdir(exist_ok=True); (ROOT / "figures").mkdir(exist_ok=True)
    metrics.to_csv(ROOT / "results/s2_condition_metrics.csv", index=False)
    summary.to_csv(ROOT / "results/s2_robustness_summary.csv", index=False)
    bootstrap = pd.DataFrame(bootstrap_rows)
    for _, index in bootstrap.groupby(["dataset", "protocol"]).groups.items():
        bootstrap.loc[index, "nll_p_holm"] = holm_adjust(bootstrap.loc[index, "nll_p"])
        bootstrap.loc[index, "accuracy_p_holm"] = holm_adjust(bootstrap.loc[index, "accuracy_p"])
    bootstrap.to_csv(ROOT / "results/s2_paired_bootstrap.csv", index=False)

    for protocol in metrics.protocol.unique():
        selected = metrics[(metrics.protocol == protocol) & (metrics.corruption_type != "clean")].copy()
        fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.7), sharex="col")
        for column, dataset in enumerate(DATASETS):
            part = selected[selected.dataset == dataset]
            for method in DISPLAY:
                chosen = part[part.method == method]
                if chosen.empty:
                    continue
                for row, metric in enumerate(("accuracy", "nll")):
                    curve = chosen.groupby("corruption_level")[metric].mean().sort_index()
                    axes[row, column].plot(curve.index, curve.values, marker="o", markersize=3,
                                           linewidth=1.25, label=DISPLAY[method], color=COLORS[DISPLAY[method]])
            axes[0, column].set_title(dataset.upper())
            axes[1, column].set_xlabel("Corruption level")
            axes[0, column].grid(alpha=.2); axes[1, column].grid(alpha=.2)
        axes[0, 0].set_ylabel("Accuracy")
        axes[1, 0].set_ylabel("NLL")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, .95))
        filename = protocol.replace("-", "_")
        fig.savefig(ROOT / f"figures/s2_{filename}_robustness.png", dpi=300, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
