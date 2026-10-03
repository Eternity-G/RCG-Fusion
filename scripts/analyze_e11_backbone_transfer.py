"""Summarize E11 and render its single publication PNG."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


DISPLAY = {"concat": "Concat", "tmc": "TMC", "qmf": "QMF",
           "pdf": "PDF", "i2moe": "I²MoE"}
STAGES = ("analytic", "anchored_router", "complete_rcg")
STAGE_DISPLAY = {"analytic": "Analytic contribution",
                 "anchored_router": "Anchored router",
                 "complete_rcg": "Complete RCG"}
COLORS = {"analytic": "#9DB7D5", "anchored_router": "#F1C17B",
          "complete_rcg": "#4F8F78"}


def ensemble_table(predictions):
    keys = ["backbone", "variant", "sample_id", "group_or_video_id", "label"]
    pcols = [column for column in predictions if column.startswith("p") and column[1:].isdigit()]
    ensemble = predictions.groupby(keys, as_index=False)[pcols].mean()
    probability = ensemble[pcols].to_numpy()
    labels = ensemble.label.to_numpy(int)
    ensemble["loss"] = -np.log(np.clip(probability[np.arange(len(labels)), labels], 1e-12, 1))
    ensemble["prediction"] = probability.argmax(1)
    ensemble["correct"] = ensemble.prediction == labels
    return ensemble, pcols


def paired_bootstrap(ensemble, repetitions=10000, seed=20261003):
    rows = []
    rng = np.random.default_rng(seed)
    for backbone, frame in ensemble.groupby("backbone"):
        base = frame[frame.variant == "base"].set_index("sample_id")
        for variant in STAGES:
            other = frame[frame.variant == variant].set_index("sample_id").loc[base.index]
            paired = pd.DataFrame({
                "group": base.group_or_video_id.to_numpy(),
                "nll_gain": base.loss.to_numpy()-other.loss.to_numpy(),
                "accuracy_gain": other.correct.to_numpy(float)-base.correct.to_numpy(float)})
            groups = paired.group.unique()
            grouped = {g: paired.loc[paired.group == g, ["nll_gain", "accuracy_gain"]].to_numpy()
                       for g in groups}
            boot = np.empty((repetitions, 2))
            for index in range(repetitions):
                sampled = rng.choice(groups, len(groups), replace=True)
                boot[index] = np.concatenate([grouped[g] for g in sampled]).mean(0)
            rows.append({"backbone": backbone, "variant": variant,
                         "ensemble_nll_gain": float(paired.nll_gain.mean()),
                         "nll_ci_low": float(np.quantile(boot[:, 0], .025)),
                         "nll_ci_high": float(np.quantile(boot[:, 0], .975)),
                         "ensemble_accuracy_gain": float(paired.accuracy_gain.mean()),
                         "accuracy_ci_low": float(np.quantile(boot[:, 1], .025)),
                         "accuracy_ci_high": float(np.quantile(boot[:, 1], .975)),
                         "groups": len(groups), "repetitions": repetitions})
    return pd.DataFrame(rows)


def render(summary, output):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": .8, "legend.frameon": False,
    })
    backbones = list(DISPLAY)
    x = np.arange(len(backbones)); width = .24
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for offset, stage in zip((-1, 0, 1), STAGES):
        values, errors = [], []
        for backbone in backbones:
            row = summary[(summary.backbone == backbone) & (summary.variant == stage)].iloc[0]
            values.append(row.nll_gain_vs_base)
            errors.append(row.nll_gain_std)
        bars = ax.bar(x+offset*width, values, width, yerr=errors, capsize=2,
                      color=COLORS[stage], edgecolor="white", linewidth=.5,
                      label=STAGE_DISPLAY[stage])
        if stage == "complete_rcg":
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+max(errors)*.08+.00015,
                        f"{value:.3f}", ha="center", va="bottom", fontsize=6,
                        color="#285C4A")
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.set_xticks(x, [DISPLAY[item] for item in backbones])
    ax.set_ylabel("NLL improvement over backbone")
    ax.set_xlabel("Coalition-valid backbone")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.14))
    ax.grid(axis="y", color="#E6E6E6", linewidth=.6)
    ax.set_axisbelow(True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/formal-e11-backbone-transfer")
    parser.add_argument("--figure", default="figures/e11_backbone_transfer.png")
    args = parser.parse_args()
    root = Path(args.run)
    metrics = pd.read_csv(root/"metrics_by_seed.csv")
    predictions = pd.read_parquet(root/"predictions.parquet")
    ensemble, _ = ensemble_table(predictions)
    bootstrap = paired_bootstrap(ensemble)
    bootstrap.to_csv(root/"ensemble_bootstrap.csv", index=False)
    summary = metrics.groupby(["backbone", "variant"]).agg(
        accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1=("macro_f1", "mean"), nll=("nll", "mean"), nll_std=("nll", "std"),
        nll_gain_vs_base=("nll_gain_vs_base", "mean"),
        nll_gain_std=("nll_gain_vs_base", "std"),
        accuracy_gain_vs_base=("accuracy_gain_vs_base", "mean"),
        clipped_harm=("clipped_harm", "mean"), correction_rate=("correction_rate", "mean"),
        negative_flip_rate=("negative_flip_rate", "mean"), mean_alpha=("mean_alpha", "mean")
    ).reset_index()
    summary = summary.merge(bootstrap, on=["backbone", "variant"], how="left")
    summary.to_csv(root/"final_summary.csv", index=False)
    figure = Path(args.figure); figure.parent.mkdir(parents=True, exist_ok=True)
    render(summary[summary.variant.isin(STAGES)], figure)
    complete = summary[summary.variant == "complete_rcg"]
    checks = {"backbones_with_positive_mean_nll_gain": int((complete.nll_gain_vs_base > 0).sum()),
              "backbones_with_nll_improvement_in_at_least_4_seeds": int(sum(
                  (metrics[(metrics.backbone == b) & (metrics.variant == "complete_rcg")]
                   .nll_gain_vs_base >= 0).sum() >= 4 for b in DISPLAY)),
              "backbones_with_positive_ensemble_nll_gain": int((complete.ensemble_nll_gain > 0).sum()),
              "claim_threshold_met": bool((complete.nll_gain_vs_base > 0).sum() >= 3)}
    (root/"checks.json").write_text(pd.Series(checks).to_json(indent=2), encoding="utf-8")
    print(complete[["backbone", "nll_gain_vs_base", "accuracy_gain_vs_base",
                    "clipped_harm", "ensemble_nll_gain", "nll_ci_low", "nll_ci_high"]]
          .to_string(index=False))
    print(checks)


if __name__ == "__main__":
    main()
