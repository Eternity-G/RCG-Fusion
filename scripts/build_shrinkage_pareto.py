"""Build the four-dataset posterior-shrinkage comparison and PNG figure.

Figure contract
---------------
Core conclusion: posterior correction improves NLL on all four datasets, but
the preregistered pi-squared adaptive rule is not consistently superior to a
fixed or selection-tuned interpolation at matched harm.
Evidence chain: one panel per dataset plots mean clipped harm against mean NLL
improvement; fixed-alpha points form the reference trade-off and the adaptive
rule is highlighted.  The companion CSV retains five-seed variability.
Archetype: quantitative grid.  Export: white-background 300 dpi PNG only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


RUNS = {
    "MOSI": "runs/rcg-posterior-shrinkage-mosi-v3",
    "MOSEI": "runs/rcg-posterior-shrinkage-mosei-v3",
    "CREMA-D": "runs/rcg-posterior-shrinkage-cremad-v1",
    "AV-MNIST": "runs/rcg-posterior-shrinkage-avmnist-v1",
}


def prediction_files(root: Path):
    return (sorted(root.glob("seed_*/predictions.parquet"))
            + sorted(root.glob("fold_*/seed_*/predictions.parquet")))


def seed_from_path(path: Path) -> int:
    return int(next(part.split("_", 1)[1] for part in path.parts
                    if part.startswith("seed_")))


def policies(full: np.ndarray, posterior: np.ndarray, pi: np.ndarray):
    classes = posterior.shape[1]
    normalized_confidence = np.clip(
        (posterior.max(1)-1/classes)/(1-1/classes), 0, 1)
    entropy = -np.sum(posterior*np.log(np.clip(posterior, 1e-12, 1)), axis=1)
    certainty = np.clip(1-entropy/np.log(classes), 0, 1)
    n = len(full)
    return {
        "Full coalition": np.zeros(n),
        "Fixed 0.25": np.full(n, .25),
        "Fixed 0.50": np.full(n, .50),
        "Fixed 0.75": np.full(n, .75),
        "Posterior confidence": .75*normalized_confidence**2,
        "Posterior entropy": .75*certainty**2,
        "Adaptive pi": .75*pi,
        "Adaptive pi^2 (ours)": .75*pi**2,
        "Adaptive pi^3": .75*pi**3,
        "Direct posterior": np.ones(n),
    }


def evaluate_frame(frame: pd.DataFrame):
    pcols = sorted((c for c in frame if c.startswith("full_p")),
                   key=lambda x: int(x[6:]))
    qcols = sorted((c for c in frame if c.startswith("posterior_p")),
                   key=lambda x: int(x[11:]))
    full = frame[pcols].to_numpy(); posterior = frame[qcols].to_numpy()
    label = frame.label.to_numpy(dtype=int); rows = np.arange(len(label))
    pi = frame.analytic_safe_probability.to_numpy()
    full_loss = -np.log(np.clip(full[rows, label], 1e-12, 1))
    full_pred = full.argmax(1)
    output = []
    for method, alpha in policies(full, posterior, pi).items():
        probability = (1-alpha[:, None])*full + alpha[:, None]*posterior
        loss = -np.log(np.clip(probability[rows, label], 1e-12, 1))
        pred = probability.argmax(1); difference = loss-full_loss
        output.append({
            "method": method, "n": len(label), "mean_alpha": alpha.mean(),
            "nll_difference": difference.mean(),
            "clipped_harm": np.minimum(np.maximum(difference, 0), 1).mean(),
            "accuracy_difference": (pred == label).mean()-(full_pred == label).mean(),
            "macro_f1_difference": (f1_score(label, pred, average="macro")
                                    - f1_score(label, full_pred, average="macro")),
            "negative_flip_rate": ((full_pred == label) & (pred != label)).mean(),
            "correction_rate": ((full_pred != label) & (pred == label)).mean(),
        })
    return output


def build(dataset: str, root: Path):
    frames = []
    for path in prediction_files(root):
        frame = pd.read_parquet(path)
        frame["_train_seed"] = seed_from_path(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for seed, frame in combined.groupby("_train_seed"):
        for item in evaluate_frame(frame):
            rows.append({"dataset": dataset, "train_seed": seed, **item})
    return rows


def add_selected_alpha(rows, dataset, root):
    path = Path(f"runs/ablation-selected-alpha-{dataset.lower().replace('-', '')}/metrics_by_seed.csv")
    if not path.exists():
        return
    for item in pd.read_csv(path).to_dict("records"):
        rows.append({
            "dataset": dataset, "train_seed": int(item["train_seed"]),
            "method": "Selection alpha", "n": int(item["n_test"]),
            "mean_alpha": item["alpha"],
            "nll_difference": item["shrink_nll"]-item["full_nll"],
            "clipped_harm": item["clipped_harm"],
            "accuracy_difference": item["shrink_accuracy"]-item["full_accuracy"],
            "macro_f1_difference": item["shrink_macro_f1"]-item["full_macro_f1"],
            "negative_flip_rate": item["negative_flip_rate"],
            "correction_rate": item["correction_rate"],
        })


def plot(summary: pd.DataFrame, output: Path):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "legend.frameon": False,
    })
    colors = {"fixed": "#6F7C85", "adaptive": "#0072B2", "main": "#D55E00",
              "other": "#8A7AAE", "full": "#333333"}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), constrained_layout=True)
    for panel, (axis, dataset) in enumerate(zip(axes, RUNS)):
        data = summary.loc[summary.dataset == dataset].copy()
        fixed = data[data.method.isin(["Fixed 0.25", "Fixed 0.50", "Fixed 0.75"])]
        axis.plot(100*fixed.clipped_harm, -fixed.nll_difference, color=colors["fixed"],
                  lw=1, marker="o", ms=3, label="Fixed alpha")
        for _, row in data.iterrows():
            if row.method in fixed.method.values or row.method == "Full coalition":
                continue
            if row.method == "Adaptive pi^2 (ours)":
                marker, color, size, z = "*", colors["main"], 65, 5
            elif row.method.startswith("Adaptive"):
                marker, color, size, z = "o", colors["adaptive"], 18, 3
            else:
                marker, color, size, z = "s", colors["other"], 15, 2
            axis.scatter(100*row.clipped_harm, -row.nll_difference, marker=marker,
                         c=color, s=size, zorder=z, edgecolors="white", linewidths=.35)
        main = data.loc[data.method == "Adaptive pi^2 (ours)"].iloc[0]
        axis.annotate(r"$\pi^2$", (100*main.clipped_harm, -main.nll_difference),
                      xytext=(3, 3), textcoords="offset points", fontsize=6,
                      color=colors["main"])
        axis.axvline(2, color="#999999", ls="--", lw=.7)
        axis.axhline(0, color="#BBBBBB", lw=.6)
        axis.set_title(dataset, fontweight="bold", pad=4)
        axis.set_xlabel("Clipped harm (%)")
        if panel == 0:
            axis.set_ylabel("NLL improvement")
        axis.text(-.13, 1.05, chr(ord("a")+panel), transform=axis.transAxes,
                  fontweight="bold", fontsize=8)
    handles = [
        mpl.lines.Line2D([], [], color=colors["fixed"], marker="o", ms=3, label="Fixed alpha"),
        mpl.lines.Line2D([], [], color=colors["adaptive"], marker="o", ls="", ms=4, label="Adaptive family"),
        mpl.lines.Line2D([], [], color=colors["main"], marker="*", ls="", ms=8, label=r"Main $\pi^2$"),
        mpl.lines.Line2D([], [], color=colors["other"], marker="s", ls="", ms=4, label="Other controls"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=4)
    fig.savefig(output/"figure_shrinkage_pareto.png", dpi=300, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/main-experiment-summary")
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, path in RUNS.items():
        rows.extend(build(dataset, Path(path))); add_selected_alpha(rows, dataset, Path(path))
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output/"shrinkage_policies_by_seed.csv", index=False)
    metrics = ["mean_alpha", "nll_difference", "clipped_harm", "accuracy_difference",
               "macro_f1_difference", "negative_flip_rate", "correction_rate"]
    summary = by_seed.groupby(["dataset", "method"], sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output/"shrinkage_policies_summary.csv", index=False)
    plot(summary.rename(columns={f"{x}_mean": x for x in metrics}), output)


if __name__ == "__main__":
    main()
