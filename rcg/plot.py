from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import MODALITIES
from .io import load_json, write_json, sha256

COLORS = {"uniform": "#8C8C8C", "confidence": "#527FA3", "confidence_calibrated": "#264D6C",
          "concat_masked": "#AC805B", "concat_clean": "#C6B39D", "tmc": "#7B6C98"}
LABELS = {"uniform": "Uniform", "confidence": "Confidence", "confidence_calibrated": "Calibrated confidence",
          "concat_masked": "Masked concat", "concat_clean": "Clean-only concat (control)", "tmc": "TMC adaptation"}


def save(fig, path):
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                        "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                        "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42,
                        "legend.frameon": False, "lines.linewidth": 1.2})


def plot(run):
    style()
    run = Path(run)
    analysis, out = run / "analysis", run / "figures"
    out.mkdir(exist_ok=True)
    meta = load_json(analysis / "analysis_metadata.json")
    dataset = load_json(run / "manifest.json")["dataset"]
    cfg = load_json(run / "manifest.json")["config"]
    bins = pd.read_csv(analysis / "reliability_bins_by_seed.csv")
    # Hero: calibrated late fusion, fixed bins; no simulated points.
    selected = bins[(bins.method == "confidence_calibrated") & (bins.score == "primary")]
    selected.to_csv(out / "reliability_contribution_source.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.205, 2.65), layout="constrained")
    for ax, m, letter in zip(axes, MODALITIES, "abc"):
        g = selected[selected.modality == m]
        for seed, sg in g.groupby("seed"):
            sg = sg.sort_values("bin")
            ax.plot(sg.r_mean, sg.u_mean, color="#AAC0D0", lw=.7, alpha=.7)
        means = g.groupby("bin")[["r_mean", "u_mean"]].mean()
        ax.plot(means.r_mean, means.u_mean, "o-", color=COLORS["confidence_calibrated"], ms=3)
        ax.axhline(0, color="0.4", lw=.7, ls="--")
        ax.set(xlabel="Calibrated reliability", title=m.capitalize(), xlim=(.48, 1.02))
        ax.text(0, 1.07, letter, transform=ax.transAxes, fontweight="bold")
    axes[0].set_ylabel("Mean deletion contribution (nats)")
    fig.suptitle(f"{dataset.upper()} | Calibrated confidence fusion | thin lines: individual training seeds", fontsize=8)
    save(fig, out / "reliability_contribution")
    summary = pd.read_csv(analysis / "diagnostics_summary.csv")
    intervals = pd.read_csv(analysis / "cluster_intervals.csv")
    support = intervals[["method", "score", "condition", "modality", "descriptive_only"]]
    summary = summary.merge(support, on=["method", "score", "condition", "modality"], how="left", validate="many_to_one")
    for kind in ("gaussian", "mask"):
        selected = summary[(summary.score == "primary") & np.isclose(summary.epsilon, .01) &
                           (summary.method.isin(["confidence", "confidence_calibrated", "concat_masked", "tmc"])) &
                           (((summary.kind == kind) & (summary.affected == summary.modality)) | (summary.kind == "clean"))]
        selected.to_csv(out / f"hcr_{kind}_source.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(7.205, 3.0), layout="constrained")
        for ax, m, letter in zip(axes, MODALITIES, "abc"):
            for method, g in selected[selected.modality == m].groupby("method"):
                g = g.sort_values("severity")
                x, y = g.severity.to_numpy(), g.hcr_mean.to_numpy()
                sd = g.hcr_std.fillna(0).to_numpy()
                ax.plot(x, y, "-", color=COLORS[method], label=LABELS[method])
                for xi, yi, low in zip(x, y, g.descriptive_only):
                    ax.plot(xi, yi, "o", ms=2.5, color=COLORS[method], markerfacecolor="white" if low else COLORS[method])
                ax.fill_between(x, np.maximum(0, y-sd), np.minimum(1, y+sd), color=COLORS[method], alpha=.12)
            ax.set(xlabel="Noise SD" if kind == "gaussian" else "Masked feature fraction", title=f"Perturbed {m}", ylim=(-.03, 1.03))
            ax.text(0, 1.07, letter, transform=ax.transAxes, fontweight="bold")
        axes[0].set_ylabel("High-confidence harm rate")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=6)
        fig.suptitle(f"{dataset.upper()} | mean ± seed SD (not CI) | open markers: low-count estimates", fontsize=8)
        save(fig, out / f"hcr_{kind}")
    selected = intervals[(intervals.kind == "clean") &
        (((intervals.method.isin(["confidence", "confidence_calibrated"])) & (intervals.score == "primary")) |
         ((intervals.method == "concat_masked") & (intervals.score == "calibrated_probe")))]
    selected.to_csv(out / "calibration_source.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.205, 2.75), layout="constrained")
    order = ["confidence", "confidence_calibrated", "concat_masked"]
    for ax, m in zip(axes, MODALITIES):
        g = selected[selected.modality == m].set_index("method")
        for i, method in enumerate(order):
            r = g.loc[method]
            if np.isfinite(r.hcr_pooled):
                ax.plot([i, i], [r.hcr_ci_low, r.hcr_ci_high], color=COLORS[method])
                ax.plot(i, r.hcr_pooled, "o", ms=4, color=COLORS[method],
                        markerfacecolor="white" if r.descriptive_only else COLORS[method])
                ax.annotate(f"n_min={int(r.n_high_min_seed)}", (i, r.hcr_pooled),
                            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=6)
        ax.set(xticks=range(3), xticklabels=["Raw\nlate", "Calibrated\nlate", "Calibrated\nprobe / concat"],
               title=m.capitalize(), ylim=(-.04, 1.12), xlim=(-.5, 2.5))
    axes[0].set_ylabel("High-confidence harm rate")
    fig.suptitle(f"{dataset.upper()} | pooled HCR, 95% video-cluster CI | open marker: descriptive only", fontsize=8)
    save(fig, out / "calibration")
    context = pd.read_parquet(analysis / "context_trajectories.parquet")
    context = context[(context.method == "confidence_calibrated") & (context.seed == cfg["seeds"][0]) &
                      (context.corruption_seed == cfg["corruption_seeds"][0])]
    fig, axes = plt.subplots(1, 3, figsize=(7.205, 2.75), layout="constrained")
    source = []
    for ax, m in zip(axes, MODALITIES):
        g = context[context.modality == m]
        priorities = g.groupby("sample_id")["switch"].any().reset_index().sort_values(["switch", "sample_id"], ascending=[False, True])
        for j, sample in enumerate(priorities.head(4).sample_id):
            sg = g[g.sample_id == sample].sort_values("severity")
            x, y = np.r_[0, sg.severity], np.r_[sg.u_clean.iloc[0], sg.u]
            ax.plot(x, y, "o-", ms=2, lw=.9, alpha=.8, label=f"Example {j+1}")
            source.append(sg)
        ax.axhline(0, ls="--", color="0.4", lw=.7)
        ax.set(xlabel="Noise SD in other modalities", title=f"Fixed {m}")
    axes[0].set_ylabel("Deletion contribution (nats)")
    fig.suptitle(f"{dataset.upper()} | selected trajectories, not prevalence | target reliability held fixed", fontsize=8)
    save(fig, out / "context_trajectories")
    pd.concat(source).to_csv(out / "context_trajectories_source.csv", index=False)
    captions = f"""# Figure captions and QA

Dataset: {dataset.upper()}. Training seeds: {meta['seeds']}. All panels use actual saved predictions.
The held-out test clips are clustered by original video. No seed or corruption repetition is a new independent video.

- Reliability–contribution: calibrated late fusion, clean test data, fixed 0.1-wide confidence bins. Thick line is the unweighted mean of seed-specific bin means; thin lines are individual seeds. Empty bins are omitted. This is descriptive and does not establish causality. Source CSV includes counts.
- HCR curves: target modality is also the perturbed modality; 0.01-nat harm tolerance. Each seed first pools the three noise/mask realizations. Lines are seed means, shading is one sample SD, not a confidence interval. Clean is evaluated once. Low-count conditions must be interpreted with cluster_intervals.csv; these curves are descriptive.
- Calibration: pooled ratios over fixed model seeds, 95% percentile intervals from {meta['bootstrap']} original-video resamples. Marked n_min is the minimum number of unique high-confidence test samples in any training seed, not number of rows. Open markers indicate insufficient high-confidence sample support. Raw/calibrated late fusion changes both probabilities and weights; calibrated concat probe changes the diagnostic score, not concat predictions.
- Context: first training and corruption seeds, four samples per modality sorted by whether a sign switch occurs, then sample ID. Examples are deliberately selected to illustrate trajectories; consult context_by_seed.csv for all-sample prevalence. The target features are unchanged.

Exports: editable SVG/PDF, lossless 600 dpi TIFF, PNG preview; white background, 183 mm width, 7 pt text. Source CSV accompanies each panel group. No significance stars, no image manipulation, no simulated scientific data.
Visual QA: review PNG previews and SVG text/clipping before publication. Journal-specific compliance has not been claimed.
"""
    (out / "CAPTIONS_AND_QA.md").write_text(captions, encoding="utf-8")
    write_json(out / "export_manifest.json", {"backend": "matplotlib", "width_mm": 183, "dpi": 600,
                                               "figures": [p.stem for p in out.glob("*.svg")],
                                               "plot_source_sha256": sha256(Path(__file__)),
                                               "analysis_metadata_sha256": sha256(analysis / "analysis_metadata.json")})
