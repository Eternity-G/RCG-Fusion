"""Actor-clustered final-system analysis for CREMA-D experiment E1."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/rcg-stable-ensemble-cremad-v2"
OUTPUT = ROOT / "runs/formal-e1-cremad/summary"
FIGURE = ROOT / "figures/e1_cremad_actor_bootstrap.png"


def probabilities(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = sorted(
        (column for column in frame if column.startswith(prefix)),
        key=lambda column: int(column.rsplit("p", 1)[1]),
    )
    return frame[columns].to_numpy(dtype=float)


def ece(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probability.max(1)
    prediction = probability.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum(np.digitize(confidence, edges[1:-1]), bins - 1)
    value = 0.0
    for index in range(bins):
        keep = indices == index
        if keep.any():
            value += keep.mean() * abs((prediction[keep] == labels[keep]).mean()
                                       - confidence[keep].mean())
    return float(value)


def metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    rows = np.arange(len(labels))
    target = np.eye(probability.shape[1])[labels]
    prediction = probability.argmax(1)
    return {
        "accuracy": float((prediction == labels).mean()),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "nll": float(-np.log(np.clip(probability[rows, labels], 1e-12, 1)).mean()),
        "brier": float(np.square(probability - target).sum(1).mean()),
        "ece": ece(probability, labels),
    }


def cluster_bootstrap(frame: pd.DataFrame, repetitions: int = 10_000,
                      seed: int = 90210) -> tuple[dict[str, list[float]], dict[str, float]]:
    labels = frame.label.to_numpy(dtype=int)
    full = probabilities(frame, "full_ensemble_p")
    final = probabilities(frame, "final_p")
    full_prediction, final_prediction = full.argmax(1), final.argmax(1)
    actors = frame.group_or_video_id.to_numpy()
    unique = np.unique(actors)
    actor_indices = {actor: np.flatnonzero(actors == actor) for actor in unique}
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(repetitions) for name in (
        "accuracy_delta", "macro_f1_delta", "nll_delta", "brier_delta",
        "ece_delta", "negative_flip_rate", "correction_rate")}
    for draw in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([actor_indices[actor] for actor in sampled])
        y, p_full, p_final = labels[index], full[index], final[index]
        full_metrics, final_metrics = metrics(p_full, y), metrics(p_final, y)
        for name in ("accuracy", "macro_f1", "nll", "brier", "ece"):
            draws[f"{name}_delta"][draw] = final_metrics[name] - full_metrics[name]
        pred_full, pred_final = p_full.argmax(1), p_final.argmax(1)
        draws["negative_flip_rate"][draw] = ((pred_full == y) & (pred_final != y)).mean()
        draws["correction_rate"][draw] = ((pred_full != y) & (pred_final == y)).mean()
    intervals = {name: np.quantile(values, [0.025, 0.975]).tolist()
                 for name, values in draws.items()}
    observed = {}
    full_metrics, final_metrics = metrics(full, labels), metrics(final, labels)
    for name in ("accuracy", "macro_f1", "nll", "brier", "ece"):
        observed[f"full_{name}"] = full_metrics[name]
        observed[f"final_{name}"] = final_metrics[name]
        observed[f"{name}_delta"] = final_metrics[name] - full_metrics[name]
    observed["negative_flip_rate"] = float(
        ((full_prediction == labels) & (final_prediction != labels)).mean())
    observed["correction_rate"] = float(
        ((full_prediction != labels) & (final_prediction == labels)).mean())
    observed["relative_error_reduction_vs_full"] = float(
        observed["accuracy_delta"] / max(1 - observed["full_accuracy"], 1e-12))
    observed["relative_nll_reduction_vs_full"] = float(
        -observed["nll_delta"] / observed["full_nll"])
    return intervals, observed


def plot(folds: pd.DataFrame) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    colors = {"Full ensemble": "#0072B2", "RCG-Fusion": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    x = np.arange(1, len(folds) + 1)
    panels = [
        ("full_ensemble_accuracy", "final_accuracy", "Accuracy",
         "(a) Accuracy across actor-held-out folds"),
        ("full_ensemble_nll", "final_nll", "NLL",
         "(b) NLL across actor-held-out folds"),
    ]
    for axis, (base_col, final_col, ylabel, title) in zip(axes, panels):
        for index, row in folds.reset_index(drop=True).iterrows():
            axis.plot([x[index] - 0.08, x[index] + 0.08],
                      [row[base_col], row[final_col]], color="#999999", linewidth=1, zorder=1)
        axis.scatter(x - 0.08, folds[base_col], s=35, color=colors["Full ensemble"],
                     label="Full ensemble", zorder=2)
        axis.scatter(x + 0.08, folds[final_col], s=35, color=colors["RCG-Fusion"],
                     label="RCG-Fusion", zorder=2)
        axis.set_xticks(x, [f"F{value + 1}" for value in folds.fold])
        axis.set_xlabel("Actor-held-out fold")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=10)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    fig.savefig(FIGURE, dpi=300, facecolor="white", transparent=False,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    paths = sorted(RUN.glob("fold_*/predictions.parquet"))
    if len(paths) != 5:
        raise ValueError(f"expected five CREMA-D folds, found {len(paths)}")
    frames = []
    for path in paths:
        fold = int(path.parent.name.split("_")[1])
        frames.append(pd.read_parquet(path).assign(fold=fold))
    frame = pd.concat(frames, ignore_index=True)
    if frame.sample_id.duplicated().any() or frame.sample_id.nunique() != 7442:
        raise ValueError("CREMA-D outer-fold predictions do not cover 7,442 unique samples")
    actor_fold_count = frame.groupby("group_or_video_id").fold.nunique()
    if len(actor_fold_count) != 91 or actor_fold_count.max() != 1:
        raise ValueError("actors are not isolated to exactly one outer test fold")

    folds = pd.read_csv(RUN / "metrics.csv")
    intervals, observed = cluster_bootstrap(frame)
    rows = []
    for metric in ("accuracy", "macro_f1", "nll", "brier", "ece"):
        rows.append({
            "metric": metric,
            "full_ensemble": observed[f"full_{metric}"],
            "rcg_fusion": observed[f"final_{metric}"],
            "delta_final_minus_full": observed[f"{metric}_delta"],
            "ci_low": intervals[f"{metric}_delta"][0],
            "ci_high": intervals[f"{metric}_delta"][1],
        })
    for metric in ("negative_flip_rate", "correction_rate"):
        rows.append({"metric": metric, "full_ensemble": np.nan,
                     "rcg_fusion": observed[metric],
                     "delta_final_minus_full": np.nan,
                     "ci_low": intervals[metric][0], "ci_high": intervals[metric][1]})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "actor_bootstrap_metrics.csv", index=False)
    summary = {
        "dataset": "CREMA-D", "samples": len(frame), "actors": 91,
        "folds": 5, "members": 5, "bootstrap_repetitions": 10_000,
        "bootstrap_cluster": "actor", "observed": observed,
        "confidence_intervals": intervals,
        "folds_improving_accuracy": int((folds.final_accuracy > folds.full_ensemble_accuracy).sum()),
        "folds_improving_nll": int((folds.final_nll < folds.full_ensemble_nll).sum()),
        "fallback_folds": int((folds.safe_shrinkage_rho == 0).sum()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(folds)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
