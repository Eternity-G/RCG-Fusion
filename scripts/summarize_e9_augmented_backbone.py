"""Summarize E9 shared dynamic-degradation backbone augmentation control."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_e9_clean_stress import (DATASETS, auc_summary, make_curves,  # noqa: E402
                                       paired_bootstrap, severity_curves,
                                       worst_modality)


def load_protocol(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for dataset, slug in DATASETS.items():
        frame = pd.read_csv(ROOT / f"runs/{prefix}-{slug}/metrics_by_condition_fold.csv")
        frame["dataset_label"] = dataset
        frames.append(frame)
    curves = make_curves(pd.concat(frames, ignore_index=True))
    severity = severity_curves(curves)
    return curves, severity, auc_summary(severity)


def plot(clean: pd.DataFrame, augmented: pd.DataFrame) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.1,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 4.7), constrained_layout=True)
    methods = [("A7_equal_ensemble", "A7 adaptive shrink"),
               ("A8_safe_fallback", "A8 safe fallback")]
    for row, (method, method_label) in enumerate(methods):
        for column, dataset in enumerate(DATASETS):
            axis = axes[row, column]
            for protocol, source, color in [("clean-trained", clean, "#E69F00"),
                                             ("augmented backbone", augmented, "#0072B2")]:
                for corruption, style in [("gaussian", "-"), ("mask", "--")]:
                    values = source[(source.dataset == dataset) & (source.method == method) &
                                    (source.corruption_type == corruption)].sort_values("severity_index")
                    axis.plot(values.severity_index, values.nll_gain_vs_full, color=color,
                              linestyle=style, marker="o", markersize=2.4, linewidth=1.15,
                              label=f"{protocol}, {corruption}")
            axis.axhline(0, color="#333333", linewidth=.7)
            axis.set_title(f"{dataset} — {method_label}", loc="left", fontsize=7.4, fontweight="bold")
            axis.set_xticks(range(5)); axis.set_xlabel("Severity index")
            if column == 0:
                axis.set_ylabel("NLL gain over matched full ensemble")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6.6)
    fig.savefig(ROOT / "figures/e9_augmented_backbone_stress.png", dpi=300,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    clean_curves, clean_severity, clean_auc = load_protocol("formal-e9-clean")
    augmented_curves, augmented_severity, augmented_auc = load_protocol("formal-e9-augmented")
    augmented_curves.to_csv(ROOT / "results/e9_augmented_stress_curves.csv", index=False)
    augmented_severity.to_csv(ROOT / "results/e9_augmented_stress_severity.csv", index=False)
    augmented_auc.to_csv(ROOT / "results/e9_augmented_stress_summary.csv", index=False)
    worst_modality(augmented_curves).to_csv(
        ROOT / "results/e9_augmented_stress_worst.csv", index=False)

    comparison = pd.concat([
        clean_auc.assign(protocol="clean-trained"),
        augmented_auc.assign(protocol="augmented-backbone"),
    ], ignore_index=True)
    comparison.to_csv(ROOT / "results/e9_protocol_comparison.csv", index=False)

    bootstrap = []
    for dataset, slug in DATASETS.items():
        for method in ["A7_equal_ensemble", "A8_safe_fallback"]:
            bootstrap.append(paired_bootstrap(
                dataset, slug, method, run_prefix="formal-e9-augmented"))
    bootstrap = pd.DataFrame(bootstrap)
    bootstrap.to_csv(ROOT / "results/e9_augmented_stress_bootstrap.csv", index=False)
    plot(clean_severity, augmented_severity)
    print("\nAugmented-backbone paired bootstrap:")
    print(bootstrap.to_string(index=False))
    full = comparison[comparison.method == "full_ensemble"].pivot(
        index=["dataset", "corruption_type"], columns="protocol", values="nll_auc").reset_index()
    full["relative_nll_auc_change"] = ((full["clean-trained"]-full["augmented-backbone"])
                                       / full["clean-trained"])
    print("\nFull-ensemble robustness change from shared augmentation:")
    print(full.to_string(index=False))


if __name__ == "__main__":
    main()
