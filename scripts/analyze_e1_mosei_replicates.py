"""Aggregate the five independent MOSEI ensembles used by experiment E1."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ROOT / "runs/rcg-stable-ensemble-mosei-v2",
    ROOT / "runs/formal-e1-mosei/g2/ensemble",
    ROOT / "runs/formal-e1-mosei/g3/ensemble",
    ROOT / "runs/formal-e1-mosei/g4/ensemble",
    ROOT / "runs/formal-e1-mosei/g5/ensemble",
]
OUTPUT = ROOT / "runs/formal-e1-mosei/summary"
FIGURE = ROOT / "figures/e1_mosei_independent_ensembles.png"


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean - half, mean + half


def probabilities(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = sorted(
        (column for column in frame if column.startswith(prefix)),
        key=lambda column: int(column.rsplit("p", 1)[1]),
    )
    return frame[columns].to_numpy(dtype=float)


def cluster_bootstrap(frames: list[pd.DataFrame], repetitions: int = 10_000,
                      seed: int = 90210) -> dict[str, list[float]]:
    reference = frames[0][["sample_id", "group_or_video_id", "label"]]
    for frame in frames[1:]:
        if not reference.equals(frame[["sample_id", "group_or_video_id", "label"]]):
            raise ValueError("replicates do not contain the same ordered MOSEI test samples")
    groups = reference["group_or_video_id"].to_numpy()
    unique = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique}
    cached = []
    for frame in frames:
        labels = frame["label"].to_numpy(dtype=int)
        full = probabilities(frame, "full_ensemble_p")
        final = probabilities(frame, "final_p")
        cached.append((labels, full, final))
    rng = np.random.default_rng(seed)
    accuracy_delta, nll_delta = np.empty(repetitions), np.empty(repetitions)
    for draw in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        acc_values, nll_values = [], []
        for labels, full, final in cached:
            y = labels[indices]
            acc_values.append((final[indices].argmax(1) == y).mean()
                              - (full[indices].argmax(1) == y).mean())
            rows = np.arange(len(indices))
            full_nll = -np.log(np.clip(full[indices][rows, y], 1e-12, 1)).mean()
            final_nll = -np.log(np.clip(final[indices][rows, y], 1e-12, 1)).mean()
            nll_values.append(final_nll - full_nll)
        accuracy_delta[draw] = np.mean(acc_values)
        nll_delta[draw] = np.mean(nll_values)
    return {
        "accuracy_delta": np.quantile(accuracy_delta, [0.025, 0.975]).tolist(),
        "nll_delta": np.quantile(nll_delta, [0.025, 0.975]).tolist(),
    }


def plot(table: pd.DataFrame) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    colors = {"Full ensemble": "#0072B2", "RCG-Fusion": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    x = np.arange(1, len(table) + 1)
    panels = [
        ("full_ensemble_accuracy", "final_accuracy", "Accuracy", "(a) Accuracy across replicates"),
        ("full_ensemble_nll", "final_nll", "NLL", "(b) NLL across replicates"),
    ]
    for axis, (base_col, final_col, ylabel, title) in zip(axes, panels):
        for index, row in table.iterrows():
            axis.plot([x[index] - 0.08, x[index] + 0.08],
                      [row[base_col], row[final_col]], color="#999999", linewidth=1, zorder=1)
        axis.scatter(x - 0.08, table[base_col], s=35, color=colors["Full ensemble"],
                     label="Full ensemble", zorder=2)
        axis.scatter(x + 0.08, table[final_col], s=35, color=colors["RCG-Fusion"],
                     label="RCG-Fusion", zorder=2)
        axis.set_xticks(x, table["replicate"])
        axis.set_xlabel("Independent five-member ensemble")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=10)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(FIGURE, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, frames = [], []
    for replicate, run in enumerate(RUNS, start=1):
        metric = pd.read_csv(run / "metrics.csv").iloc[0].to_dict()
        metric["replicate"] = f"G{replicate}"
        rows.append(metric)
        frames.append(pd.read_parquet(run / "fold_0/predictions.parquet"))
    table = pd.DataFrame(rows)
    table["accuracy_delta_pp"] = 100 * (
        table["final_accuracy"] - table["full_ensemble_accuracy"])
    table["nll_delta"] = table["final_nll"] - table["full_ensemble_nll"]
    table.to_csv(OUTPUT / "metrics_by_replicate.csv", index=False)

    accuracy_delta = table["final_accuracy"].to_numpy() - table["full_ensemble_accuracy"].to_numpy()
    nll_delta = table["final_nll"].to_numpy() - table["full_ensemble_nll"].to_numpy()
    summary = {
        "replicates": len(table),
        "members_per_replicate": 5,
        "final_accuracy_mean": float(table.final_accuracy.mean()),
        "final_accuracy_sd": float(table.final_accuracy.std(ddof=1)),
        "full_accuracy_mean": float(table.full_ensemble_accuracy.mean()),
        "full_accuracy_sd": float(table.full_ensemble_accuracy.std(ddof=1)),
        "accuracy_delta_pp_mean": float(100 * accuracy_delta.mean()),
        "accuracy_delta_pp_sd": float(100 * accuracy_delta.std(ddof=1)),
        "accuracy_delta_t_ci_pp": [100 * value for value in mean_ci(accuracy_delta)],
        "accuracy_delta_t_p": float(ttest_1samp(accuracy_delta, 0).pvalue),
        "final_nll_mean": float(table.final_nll.mean()),
        "final_nll_sd": float(table.final_nll.std(ddof=1)),
        "full_nll_mean": float(table.full_ensemble_nll.mean()),
        "full_nll_sd": float(table.full_ensemble_nll.std(ddof=1)),
        "nll_delta_mean": float(nll_delta.mean()),
        "nll_delta_sd": float(nll_delta.std(ddof=1)),
        "nll_delta_t_ci": list(mean_ci(nll_delta)),
        "nll_delta_t_p": float(ttest_1samp(nll_delta, 0).pvalue),
        "system_error_reduction_mean": float(table.relative_error_reduction_vs_mean_single.mean()),
        "system_error_reduction_sd": float(table.relative_error_reduction_vs_mean_single.std(ddof=1)),
        "replicates_above_10pct_system_error_reduction": int(
            (table.relative_error_reduction_vs_mean_single >= 0.10).sum()),
        "replicates_improving_accuracy_vs_full": int((accuracy_delta > 0).sum()),
        "replicates_improving_nll_vs_full": int((nll_delta < 0).sum()),
        "cluster_bootstrap_ci": cluster_bootstrap(frames),
        "bootstrap_repetitions": 10_000,
        "bootstrap_cluster": "original MOSEI video",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(table)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

