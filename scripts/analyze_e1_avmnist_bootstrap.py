"""Class-stratified final-system analysis for AV-MNIST experiment E1."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/rcg-stable-ensemble-avmnist-v2"
OUTPUT = ROOT / "runs/formal-e1-avmnist/summary"
FIGURE = ROOT / "figures/e1_avmnist_stratified_bootstrap.png"


def probabilities(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    columns = sorted(
        (column for column in frame if column.startswith(prefix)),
        key=lambda column: int(column.rsplit("p", 1)[1]),
    )
    if len(columns) != 10:
        raise ValueError(f"expected 10 probability columns for {prefix}, found {len(columns)}")
    values = frame[columns].to_numpy(dtype=float)
    if not np.allclose(values.sum(1), 1.0, atol=1e-5):
        raise ValueError(f"{prefix} probabilities do not sum to one")
    return values


def ece(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probability.max(1)
    prediction = probability.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum(np.digitize(confidence, edges[1:-1]), bins - 1)
    value = 0.0
    for index in range(bins):
        keep = indices == index
        if keep.any():
            value += keep.mean() * abs(
                (prediction[keep] == labels[keep]).mean() - confidence[keep].mean()
            )
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


def stratified_bootstrap(
    frame: pd.DataFrame,
    repetitions: int = 10_000,
    seed: int = 90210,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    labels = frame.label.to_numpy(dtype=int)
    full = probabilities(frame, "full_ensemble_p")
    final = probabilities(frame, "final_p")
    classes = np.unique(labels)
    class_indices = {label: np.flatnonzero(labels == label) for label in classes}
    rng = np.random.default_rng(seed)
    names = (
        "accuracy_delta", "macro_f1_delta", "nll_delta", "brier_delta",
        "ece_delta", "negative_flip_rate", "correction_rate",
    )
    draws = {name: np.empty(repetitions) for name in names}
    for draw in range(repetitions):
        index = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices.values()
        ])
        y, p_full, p_final = labels[index], full[index], final[index]
        full_metrics, final_metrics = metrics(p_full, y), metrics(p_final, y)
        for name in ("accuracy", "macro_f1", "nll", "brier", "ece"):
            draws[f"{name}_delta"][draw] = final_metrics[name] - full_metrics[name]
        pred_full, pred_final = p_full.argmax(1), p_final.argmax(1)
        draws["negative_flip_rate"][draw] = ((pred_full == y) & (pred_final != y)).mean()
        draws["correction_rate"][draw] = ((pred_full != y) & (pred_final == y)).mean()

    intervals = {
        name: np.quantile(values, [0.025, 0.975]).tolist()
        for name, values in draws.items()
    }
    full_metrics, final_metrics = metrics(full, labels), metrics(final, labels)
    observed: dict[str, float] = {}
    for name in ("accuracy", "macro_f1", "nll", "brier", "ece"):
        observed[f"full_{name}"] = full_metrics[name]
        observed[f"final_{name}"] = final_metrics[name]
        observed[f"{name}_delta"] = final_metrics[name] - full_metrics[name]
    pred_full, pred_final = full.argmax(1), final.argmax(1)
    observed["negative_flip_rate"] = float(
        ((pred_full == labels) & (pred_final != labels)).mean()
    )
    observed["correction_rate"] = float(
        ((pred_full != labels) & (pred_final == labels)).mean()
    )
    observed["relative_error_reduction_vs_full"] = float(
        observed["accuracy_delta"] / max(1 - observed["full_accuracy"], 1e-12)
    )
    observed["relative_nll_reduction_vs_full"] = float(
        -observed["nll_delta"] / observed["full_nll"]
    )
    return intervals, observed


def class_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame.label.to_numpy(dtype=int)
    full_prediction = probabilities(frame, "full_ensemble_p").argmax(1)
    final_prediction = probabilities(frame, "final_p").argmax(1)
    full_recall = recall_score(labels, full_prediction, labels=np.arange(10), average=None)
    final_recall = recall_score(labels, final_prediction, labels=np.arange(10), average=None)
    rows = []
    for label in range(10):
        keep = labels == label
        rows.append({
            "digit": label,
            "samples": int(keep.sum()),
            "full_ensemble_recall": float(full_recall[label]),
            "rcg_fusion_recall": float(final_recall[label]),
            "recall_delta": float(final_recall[label] - full_recall[label]),
            "negative_flips": int(((full_prediction == labels) & (final_prediction != labels) & keep).sum()),
            "corrections": int(((full_prediction != labels) & (final_prediction == labels) & keep).sum()),
        })
    return pd.DataFrame(rows)


def plot(class_frame: pd.DataFrame) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    blue, orange = "#0072B2", "#D55E00"
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    x = class_frame.digit.to_numpy()
    for _, row in class_frame.iterrows():
        axes[0].plot(
            [row.digit - 0.08, row.digit + 0.08],
            [row.full_ensemble_recall, row.rcg_fusion_recall],
            color="#999999", linewidth=1, zorder=1,
        )
    axes[0].scatter(x - 0.08, class_frame.full_ensemble_recall, s=34, color=blue,
                    label="Full ensemble", zorder=2)
    axes[0].scatter(x + 0.08, class_frame.rcg_fusion_recall, s=34, color=orange,
                    label="RCG-Fusion", zorder=2)
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Digit class")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("(a) Recall by class", loc="left", fontsize=10)
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False, loc="lower left")

    width = 0.36
    axes[1].bar(x - width / 2, class_frame.corrections, width, color="#009E73",
                label="Incorrect to correct")
    axes[1].bar(x + width / 2, class_frame.negative_flips, width, color="#CC79A7",
                label="Correct to incorrect")
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Digit class")
    axes[1].set_ylabel("Number of samples")
    axes[1].set_title("(b) Paired prediction transitions", loc="left", fontsize=10)
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[1].legend(frameon=False, loc="upper right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURE, dpi=300, facecolor="white", transparent=False,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    paths = sorted(RUN.glob("fold_*/predictions.parquet"))
    if len(paths) != 1:
        raise ValueError(f"expected one AV-MNIST test prediction file, found {len(paths)}")
    frame = pd.read_parquet(paths[0])
    if len(frame) != 3600 or frame.sample_id.nunique() != 3600:
        raise ValueError("AV-MNIST predictions do not cover 3,600 unique test samples")
    if set(frame.label.unique()) != set(range(10)):
        raise ValueError("AV-MNIST predictions do not contain all ten digit classes")

    intervals, observed = stratified_bootstrap(frame)
    class_frame = class_statistics(frame)
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
        rows.append({
            "metric": metric,
            "full_ensemble": np.nan,
            "rcg_fusion": observed[metric],
            "delta_final_minus_full": np.nan,
            "ci_low": intervals[metric][0],
            "ci_high": intervals[metric][1],
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "stratified_bootstrap_metrics.csv", index=False)
    class_frame.to_csv(OUTPUT / "class_metrics.csv", index=False)
    metrics_record = json.loads((RUN / "fold_0/metrics.json").read_text(encoding="utf-8"))
    summary = {
        "dataset": "AV-MNIST",
        "samples": len(frame),
        "classes": 10,
        "members": int(metrics_record["members"]),
        "safe_shrinkage_rho": float(metrics_record["safe_shrinkage_rho"]),
        "bootstrap_repetitions": 10_000,
        "bootstrap_scheme": "within-class stratified sample bootstrap",
        "observed": observed,
        "confidence_intervals": intervals,
        "classes_improving_recall": int((class_frame.recall_delta > 0).sum()),
        "classes_declining_recall": int((class_frame.recall_delta < 0).sum()),
        "total_corrections": int(class_frame.corrections.sum()),
        "total_negative_flips": int(class_frame.negative_flips.sum()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(class_frame)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
