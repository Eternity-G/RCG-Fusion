"""Analyze five independent P0-v2 ensemble groups for MOSI and MOSEI."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, t

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("g1", "g2", "g3", "g4", "g5")


def columns(frame, prefix="p"):
    values = [column for column in frame if column.startswith(prefix)]
    if prefix == "p": values = [column for column in values if column[1:].isdigit()]
    else: values = [column for column in values if column[len(prefix):].isdigit()]
    return sorted(values, key=lambda value: int(value[len(prefix):]))


def load(dataset, group):
    path = (ROOT/f"runs/formal-e8-{dataset}/final_predictions.parquet" if group == "g1"
            else ROOT/f"runs/formal-s1-replicates/{dataset}/{group}/predictions.parquet")
    frame = pd.read_parquet(path).copy(); frame.sample_id = frame.sample_id.astype(str)
    return frame.sort_values(["fold", "sample_id"]).reset_index(drop=True)


def metrics(frame, group):
    labels = frame.label.to_numpy(int); rows = np.arange(len(labels))
    final = frame[columns(frame)].to_numpy(float)
    full = frame[columns(frame, "full_p")].to_numpy(float)
    return {"dataset": frame.dataset.iloc[0], "group": group,
            "full_accuracy": float((full.argmax(1) == labels).mean()),
            "final_accuracy": float((final.argmax(1) == labels).mean()),
            "accuracy_gain": float(((final.argmax(1) == labels).astype(float)-
                                    (full.argmax(1) == labels).astype(float)).mean()),
            "full_nll": float(-np.log(np.clip(full[rows, labels], 1e-12, 1)).mean()),
            "final_nll": float(-np.log(np.clip(final[rows, labels], 1e-12, 1)).mean()),
            "nll_gain": float((np.log(np.clip(final[rows, labels], 1e-12, 1))-
                               np.log(np.clip(full[rows, labels], 1e-12, 1))).mean())}


def training_ci(values):
    values = np.asarray(values, float); mean = values.mean()
    margin = t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
    return mean, mean-margin, mean+margin, float(ttest_1samp(values, 0).pvalue)


def cluster_bootstrap(frames, repetitions=10_000, seed=20261009):
    base = frames[0][["fold", "sample_id", "group_or_video_id", "label"]].copy()
    values = []
    for frame in frames:
        if not np.array_equal(base.sample_id.to_numpy(), frame.sample_id.to_numpy()):
            raise AssertionError("replicate sample order differs")
        labels = frame.label.to_numpy(int); rows = np.arange(len(labels))
        final = frame[columns(frame)].to_numpy(float); full = frame[columns(frame, "full_p")].to_numpy(float)
        values.append(np.stack([
            np.log(np.clip(final[rows, labels], 1e-12, 1))-np.log(np.clip(full[rows, labels], 1e-12, 1)),
            (final.argmax(1) == labels).astype(float)-(full.argmax(1) == labels).astype(float)], 1))
    values = np.mean(values, axis=0); groups = base.group_or_video_id.astype(str).to_numpy()
    unique = np.unique(groups); index = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed); draws = np.empty((repetitions, 2))
    for draw in range(repetitions):
        sample = rng.choice(unique, len(unique), replace=True)
        take = np.concatenate([index[group] for group in sample]); draws[draw] = values[take].mean(0)
    return {"nll_ci_low": float(np.quantile(draws[:, 0], .025)),
            "nll_ci_high": float(np.quantile(draws[:, 0], .975)),
            "accuracy_ci_low": float(np.quantile(draws[:, 1], .025)),
            "accuracy_ci_high": float(np.quantile(draws[:, 1], .975))}


def main():
    rows, summaries = [], []
    for dataset in ("mosi", "mosei"):
        frames = [load(dataset, group) for group in GROUPS]
        table = pd.DataFrame([metrics(frame, group) for frame, group in zip(frames, GROUPS)])
        rows.append(table); bootstrap = cluster_bootstrap(frames)
        acc = training_ci(table.accuracy_gain); nll = training_ci(table.nll_gain)
        summaries.append({"dataset": dataset, "accuracy_gain_mean": acc[0],
                          "accuracy_training_ci_low": acc[1], "accuracy_training_ci_high": acc[2],
                          "accuracy_training_p": acc[3], "nll_gain_mean": nll[0],
                          "nll_training_ci_low": nll[1], "nll_training_ci_high": nll[2],
                          "nll_training_p": nll[3], **bootstrap,
                          "accuracy_wins": int((table.accuracy_gain > 0).sum()),
                          "nll_wins": int((table.nll_gain > 0).sum())})
    rows = pd.concat(rows, ignore_index=True); summary = pd.DataFrame(summaries)
    rows.to_csv(ROOT/"results/s1_replicates_by_group.csv", index=False)
    summary.to_csv(ROOT/"results/s1_replicates_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.1), constrained_layout=True)
    colors = {"mosi": "#0072B2", "mosei": "#D55E00"}
    for dataset in ("mosi", "mosei"):
        data = rows[rows.dataset == dataset]; x = np.arange(1, 6)
        axes[0].plot(x, 100*data.accuracy_gain, "o-", color=colors[dataset], label=dataset.upper())
        axes[1].plot(x, data.nll_gain, "o-", color=colors[dataset], label=dataset.upper())
    for axis in axes:
        axis.axhline(0, color="black", linewidth=.8); axis.set_xticks(np.arange(1, 6))
        axis.set_xlabel("Independent ensemble group"); axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Accuracy gain (pp)"); axes[0].set_title("a  Decision gain", loc="left", fontweight="bold")
    axes[1].set_ylabel("NLL reduction"); axes[1].set_title("b  Probability gain", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    fig.savefig(ROOT/"figures/s1_independent_replicates.png", dpi=300, facecolor="white", transparent=False)
    plt.close(fig)
    print(rows.to_string(index=False)); print(summary.to_string(index=False))


if __name__ == "__main__": main()
