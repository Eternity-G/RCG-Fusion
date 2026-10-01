"""Formal E3 contribution-identification evaluation on frozen test predictions.

The script does not fit or select a model. Test labels are read only to compute
pre-registered evaluation metrics after every score has been produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import spearmanr, ttest_rel
from sklearn.metrics import average_precision_score, roc_auc_score

from rcg.posterior_analytic import analytic_contribution
from rcg.posterior_analytic_pipeline import (_load_backbone, calibrate_members,
                                               fit_ensemble_temperature)
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import load_dataset, predict_outputs


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = (11, 22, 33, 44, 55)
TAU = 0.01
CONFIGS = {
    "mosi": {
        "base": ROOT / "runs/rcg-fusion-mosi-v4",
        "posterior": ROOT / "runs/rcg-posterior-analytic-mosi-v2b",
        "oof": ROOT / "runs/rcg-fusion-mosi-v4/fold_0/oof_targets.npz",
    },
    "mosei": {
        "base": ROOT / "runs/rcg-fusion-mosei-shrinkage-v1",
        "posterior": ROOT / "runs/rcg-posterior-analytic-mosei-shrinkage-v1",
        "oof": ROOT / "runs/formal-e2-mosei-oof/oof_targets.npz",
    },
    "cremad": {
        "base": ROOT / "runs/rcg-fusion-cremad-v1",
        "posterior": ROOT / "runs/rcg-posterior-analytic-cremad-v1",
        "oof": ROOT / "runs/formal-e2-cremad-oof/fold_{fold}/oof_targets.npz",
    },
    "avmnist": {
        "base": ROOT / "runs/rcg-fusion-avmnist-v1",
        "posterior": ROOT / "runs/rcg-posterior-analytic-avmnist-v1",
        "oof": ROOT / "runs/formal-e2-avmnist-oof/oof_targets.npz",
    },
}
METHODS = {
    "max_confidence": "Max probability",
    "negative_entropy": "Negative entropy",
    "direct_risk_regression": "Direct risk regression",
    "full_posterior_analytic": "Full-output posterior",
    "plain_mlp_posterior": "Plain MLP posterior",
    "learned_posterior_analytic": "Learned posterior analytic",
}
STRONG_METHODS = {
    "concat": "Max probability",
    "tmc": "TMC evidence",
    "qmf": "QMF quality",
    "pdf": "PDF Co-Belief",
    "i2moe": "I²MoE routing",
}
POSTERIOR_SEEDS = (11, 22, 33, 44, 55)


class PlainPosteriorMLP(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(.2),
            nn.Linear(128, classes),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def posterior_features(split: dict, probabilities: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [*split["x"], probabilities.reshape(len(probabilities), -1)], axis=1
    ).astype(np.float32)


def train_plain_member(train_x: np.ndarray, train_y: np.ndarray,
                       selection_x: np.ndarray, selection_y: np.ndarray,
                       classes: int, seed: int, device: str) -> PlainPosteriorMLP:
    torch.manual_seed(seed); np.random.seed(seed)
    model = PlainPosteriorMLP(train_x.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tx = torch.as_tensor(train_x, device=device)
    ty = torch.as_tensor(train_y, dtype=torch.long, device=device)
    sx = torch.as_tensor(selection_x, device=device)
    sy = torch.as_tensor(selection_y, dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(100):
        model.train()
        for indices in torch.randperm(len(ty), generator=generator, device=device).split(256):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tx[indices]); probability = logits.softmax(-1)
            target = F.one_hot(ty[indices], classes).to(probability.dtype)
            loss = F.cross_entropy(logits, ty[indices]) + .1 * (
                (probability-target).square().sum(-1).mean()
            )
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            selection_nll = float(F.cross_entropy(model(sx), sy))
        if selection_nll < best - 1e-6:
            best = selection_nll
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    if best_state is None:
        raise RuntimeError("plain posterior MLP produced no checkpoint")
    model.load_state_dict(best_state); model.eval()
    return model


def plain_mlp_posterior(dataset: str, fold_index: int, task_seed: int,
                        splits: dict, selection_probability: np.ndarray,
                        test_probability: np.ndarray, classes: int, device: str
                        ) -> np.ndarray:
    cache = (ROOT / "runs/formal-e3-plain-mlp" / dataset
             / f"fold_{fold_index}/seed_{task_seed}.npz")
    if cache.exists():
        with np.load(cache) as saved:
            return saved["posterior"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    oof_spec = CONFIGS[dataset]["oof"]
    with np.load(Path(str(oof_spec).format(fold=fold_index))) as saved:
        oof_probability = saved["probabilities"].mean(0)
    train_x = posterior_features(splits["train"], oof_probability)
    selection_x = posterior_features(splits["selection"], selection_probability)
    test_x = posterior_features(splits["test"], test_probability)
    members, selection_members = [], []
    for posterior_seed in POSTERIOR_SEEDS:
        model = train_plain_member(
            train_x, splits["train"]["y"], selection_x,
            splits["selection"]["y"], classes,
            task_seed * 1000 + posterior_seed, device,
        )
        with torch.inference_mode():
            selection_members.append(model(torch.as_tensor(
                selection_x, device=device)).softmax(-1).cpu().numpy())
            members.append(model(torch.as_tensor(
                test_x, device=device)).softmax(-1).cpu().numpy())
        del model
    temperature = fit_ensemble_temperature(
        selection_members, splits["selection"]["y"]
    )
    posterior = np.mean(calibrate_members(members, temperature), axis=0)
    np.savez_compressed(cache, posterior=posterior.astype(np.float32),
                        temperature=np.asarray(temperature, dtype=np.float32))
    return posterior


def entropy(probability: np.ndarray) -> np.ndarray:
    value = np.clip(probability, 1e-8, 1.0)
    return -(value * np.log(value)).sum(-1)


def ece(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if selected.any():
            result += selected.mean() * abs(probability[selected].mean() - target[selected].mean())
    return float(result)


def score_metrics(score: np.ndarray, value_score: np.ndarray,
                  benefit: np.ndarray, probabilistic: bool) -> dict[str, float]:
    safe = benefit > TAU
    if len(np.unique(safe)) < 2:
        return {key: np.nan for key in (
            "auroc", "auprc", "spearman", "top10_precision",
            "top20_precision", "brier", "ece", "safe_prevalence",
        )}
    top10 = score >= np.quantile(score, 0.9)
    top20 = score >= np.quantile(score, 0.8)
    return {
        "auroc": float(roc_auc_score(safe, score)),
        "auprc": float(average_precision_score(safe, score)),
        "spearman": float(spearmanr(value_score, benefit).statistic),
        "top10_precision": float(safe[top10].mean()),
        "top20_precision": float(safe[top20].mean()),
        "brier": float(np.mean((score - safe) ** 2)) if probabilistic else np.nan,
        "ece": ece(score, safe) if probabilistic else np.nan,
        "safe_prevalence": float(safe.mean()),
    }


def direct_predicted_losses(path: Path, names: list[str], sample_ids: np.ndarray
                            ) -> np.ndarray:
    frame = pd.read_parquet(path)
    if frame.sample_id.astype(str).tolist() != np.asarray(sample_ids).astype(str).tolist():
        raise ValueError(f"sample order differs in {path}")
    return np.stack([frame[f"predicted_loss_{name}"].to_numpy() for name in names], 1)


def evaluate_dataset(dataset: str, device: str
                     ) -> tuple[list[dict], list[dict]]:
    config = CONFIGS[dataset]
    folds, modality_names = load_dataset(dataset, ROOT / "data")
    masks = nonempty_coalitions(len(modality_names))
    coalition_names = [
        "+".join(name for name, keep in zip(modality_names, mask) if keep)
        for mask in masks
    ]
    full = len(masks) - 1
    metric_rows, bin_rows = [], []
    for seed in SEEDS:
        accumulated = {key: [] for key in (
            "benefit", "confidence", "negative_entropy", "direct",
            "full_probability", "full_expected", "learned_probability",
            "learned_expected", "plain_probability", "plain_expected",
        )}
        for fold_index, splits in enumerate(folds):
            test = splits["test"]
            model, temperatures, _, _ = _load_backbone(
                config["base"] / f"fold_{fold_index}/seed_{seed}/backbone.pt", device
            )
            output = predict_outputs(model, test["x"], device, temperatures)
            probability = output["probabilities"]
            selection_probability = predict_outputs(
                model, splits["selection"]["x"], device, temperatures
            )["probabilities"]
            losses = -np.log(np.clip(
                np.take_along_axis(
                    probability, test["y"][:, None, None], axis=2
                ).squeeze(2),
                1e-8, 1.0,
            ))
            benefit = losses[:, full, None] - losses
            predicted_losses = direct_predicted_losses(
                config["base"] / f"fold_{fold_index}/seed_{seed}/predictions.parquet",
                coalition_names, test["id"],
            )
            direct = predicted_losses[:, full, None] - predicted_losses
            with np.load(
                config["posterior"] / f"fold_{fold_index}/seed_{seed}/analytic_outputs.npz"
            ) as saved:
                learned_probability = saved["safe_probability"]
                learned_expected = saved["expected_benefit"]
            full_posterior = probability[:, full]
            full_analytic = analytic_contribution(
                torch.as_tensor(full_posterior, dtype=torch.float64),
                torch.as_tensor(probability, dtype=torch.float64), tau=TAU,
            )
            plain_posterior = plain_mlp_posterior(
                dataset, fold_index, seed, splits, selection_probability,
                probability, probability.shape[-1], device,
            )
            plain_analytic = analytic_contribution(
                torch.as_tensor(plain_posterior, dtype=torch.float64),
                torch.as_tensor(probability, dtype=torch.float64), tau=TAU,
            )
            values = {
                "benefit": benefit,
                "confidence": probability.max(-1),
                "negative_entropy": 1 - entropy(probability) / np.log(probability.shape[-1]),
                "direct": direct,
                "full_probability": full_analytic["safe_probability"].numpy(),
                "full_expected": full_analytic["expected_benefit"].numpy(),
                "learned_probability": learned_probability,
                "learned_expected": learned_expected,
                "plain_probability": plain_analytic["safe_probability"].numpy(),
                "plain_expected": plain_analytic["expected_benefit"].numpy(),
            }
            for key, value in values.items():
                accumulated[key].append(value)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        arrays = {key: np.concatenate(value, axis=0) for key, value in accumulated.items()}
        for coalition_index, coalition_name in enumerate(coalition_names[:-1]):
            benefit = arrays["benefit"][:, coalition_index]
            method_scores = {
                "max_confidence": (
                    arrays["confidence"][:, coalition_index],
                    arrays["confidence"][:, coalition_index], False,
                ),
                "negative_entropy": (
                    arrays["negative_entropy"][:, coalition_index],
                    arrays["negative_entropy"][:, coalition_index], False,
                ),
                "direct_risk_regression": (
                    arrays["direct"][:, coalition_index],
                    arrays["direct"][:, coalition_index], False,
                ),
                "full_posterior_analytic": (
                    arrays["full_probability"][:, coalition_index],
                    arrays["full_expected"][:, coalition_index], True,
                ),
                "plain_mlp_posterior": (
                    arrays["plain_probability"][:, coalition_index],
                    arrays["plain_expected"][:, coalition_index], True,
                ),
                "learned_posterior_analytic": (
                    arrays["learned_probability"][:, coalition_index],
                    arrays["learned_expected"][:, coalition_index], True,
                ),
            }
            for method, (event_score, value_score, probabilistic) in method_scores.items():
                metric_rows.append({
                    "dataset": dataset, "train_seed": seed,
                    "coalition": coalition_name, "method": method,
                    "n": len(benefit),
                    **score_metrics(event_score, value_score, benefit, probabilistic),
                })

        learned_score = arrays["learned_probability"][:, :-1].ravel()
        learned_benefit = arrays["benefit"][:, :-1].ravel()
        order = np.argsort(learned_score, kind="stable")
        for decile, indices in enumerate(np.array_split(order, 10), start=1):
            bin_rows.append({
                "dataset": dataset, "train_seed": seed, "decile": decile,
                "mean_predicted_probability": float(learned_score[indices].mean()),
                "observed_safe_rate": float((learned_benefit[indices] > TAU).mean()),
                "mean_observed_benefit": float(learned_benefit[indices].mean()),
                "n": len(indices),
            })
    return metric_rows, bin_rows


def evaluate_native_scores() -> pd.DataFrame:
    rows = []
    root = ROOT / "runs/strong-observation"
    for dataset in ("mosi", "mosei", "cremad"):
        dataset_root = root / dataset
        for method, label in STRONG_METHODS.items():
            by_seed: dict[int, list[dict]] = {seed: [] for seed in SEEDS}
            for fold_path in sorted(dataset_root.glob("fold_*")):
                for seed in SEEDS:
                    path = fold_path / method / f"seed_{seed}/samples.parquet"
                    frame = pd.read_parquet(path)
                    modalities = [
                        column.removeprefix("native_score_")
                        for column in frame.columns if column.startswith("native_score_")
                    ]
                    for modality in modalities:
                        # Removing m is beneficial iff its full-alliance deletion
                        # contribution is negative beyond the fixed tolerance.
                        benefit = -frame[f"deletion_contribution_{modality}"].to_numpy()
                        score = -frame[f"native_score_{modality}"].to_numpy()
                        metrics = score_metrics(score, score, benefit, False)
                        by_seed[seed].append(metrics)
            for seed, values in by_seed.items():
                rows.append({
                    "dataset": dataset, "train_seed": seed,
                    "method": method, "method_label": label,
                    **{
                        metric: float(np.nanmean([value[metric] for value in values]))
                        for metric in ("auroc", "auprc", "spearman",
                                       "top10_precision", "top20_precision",
                                       "safe_prevalence")
                    },
                })
    return pd.DataFrame(rows)


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    per_seed = metrics.groupby(
        ["dataset", "train_seed", "method"], as_index=False
    )[["auroc", "auprc", "spearman", "top10_precision", "top20_precision",
       "brier", "ece", "safe_prevalence"]].mean()
    summary = per_seed.groupby(["dataset", "method"], as_index=False).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"),
        spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
        top10_precision_mean=("top10_precision", "mean"),
        top20_precision_mean=("top20_precision", "mean"),
        brier_mean=("brier", "mean"), ece_mean=("ece", "mean"),
        safe_prevalence=("safe_prevalence", "mean"),
    )
    return per_seed, summary


def paired_comparisons(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    learned_name = "learned_posterior_analytic"
    for dataset in CONFIGS:
        current = per_seed[per_seed.dataset == dataset]
        for baseline in METHODS:
            if baseline == learned_name:
                continue
            for metric in ("auroc", "auprc", "spearman", "top20_precision",
                           "brier", "ece"):
                pivot = current.pivot(index="train_seed", columns="method", values=metric)
                difference = pivot[learned_name] - pivot[baseline]
                if difference.isna().all():
                    continue
                valid = difference.dropna()
                rows.append({
                    "dataset": dataset, "baseline": baseline, "metric": metric,
                    "learned_minus_baseline": float(valid.mean()),
                    "learned_better_seeds": int((
                        valid < 0 if metric in {"brier", "ece"} else valid > 0
                    ).sum()),
                    "n_seeds": len(valid),
                    "paired_t_pvalue": (
                        np.nan if np.allclose(valid, 0) else
                        float(ttest_rel(
                            pivot.loc[valid.index, learned_name],
                            pivot.loc[valid.index, baseline],
                        ).pvalue)
                    ),
                })
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, bins: pd.DataFrame,
                native: pd.DataFrame, path: Path) -> None:
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    datasets = list(CONFIGS)
    methods = list(METHODS)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.5), constrained_layout=True)
    x = np.arange(len(datasets)); width = 0.125
    for index, method in enumerate(methods):
        selected = summary[summary.method == method].set_index("dataset").loc[datasets]
        axes[0, 0].bar(
            x + (index - (len(methods)-1)/2) * width, selected.auroc_mean, width,
            yerr=selected.auroc_std, capsize=2, color=colors[index],
            label=METHODS[method],
        )
        axes[0, 1].bar(
            x + (index - (len(methods)-1)/2) * width, selected.spearman_mean, width,
            yerr=selected.spearman_std, capsize=2, color=colors[index],
        )
    for axis, ylabel, title in (
        (axes[0, 0], "Macro AUROC", "(a) Beneficial-coalition identification"),
        (axes[0, 1], "Macro Spearman", "(b) Benefit ranking"),
    ):
        axis.set_xticks(x, [name.upper() for name in datasets])
        axis.set_ylabel(ylabel); axis.set_title(title, loc="left", fontsize=10)
        axis.grid(axis="y", color="#dddddd", linewidth=.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    bin_summary = bins.groupby(["dataset", "decile"], as_index=False).agg(
        observed=("observed_safe_rate", "mean"),
        observed_std=("observed_safe_rate", "std"),
    )
    for index, dataset in enumerate(datasets):
        selected = bin_summary[bin_summary.dataset == dataset]
        axes[1, 0].plot(selected.decile, selected.observed, marker="o",
                        color=colors[index], label=dataset.upper())
        axes[1, 0].fill_between(
            selected.decile, selected.observed-selected.observed_std,
            selected.observed+selected.observed_std, color=colors[index], alpha=.12,
        )
    axes[1, 0].set_xlabel("Predicted-benefit probability decile")
    axes[1, 0].set_ylabel("Observed beneficial rate")
    axes[1, 0].set_title("(c) Learned analytic score stratification", loc="left", fontsize=10)
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(axis="y", color="#dddddd", linewidth=.7)
    axes[1, 0].spines[["top", "right"]].set_visible(False)

    native_summary = native.groupby(["dataset", "method_label"]).auroc.mean().unstack()
    native_summary = native_summary.loc[["mosi", "mosei", "cremad"]]
    image = axes[1, 1].imshow(native_summary.to_numpy(), vmin=.35, vmax=.75,
                              cmap="viridis", aspect="auto")
    axes[1, 1].set_xticks(np.arange(len(native_summary.columns)),
                          native_summary.columns, rotation=25, ha="right")
    axes[1, 1].set_yticks(np.arange(len(native_summary.index)),
                          [value.upper() for value in native_summary.index])
    axes[1, 1].set_title("(d) Native scores: beneficial removal AUROC", loc="left", fontsize=10)
    for row in range(native_summary.shape[0]):
        for column in range(native_summary.shape[1]):
            value = native_summary.iloc[row, column]
            axes[1, 1].text(column, row, f"{value:.2f}", ha="center", va="center",
                            color="white" if value < .55 else "black", fontsize=8)
    fig.colorbar(image, ax=axes[1, 1], fraction=.046, pad=.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white", transparent=False,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_rows, bin_rows = [], []
    for dataset in CONFIGS:
        current_metrics, current_bins = evaluate_dataset(dataset, device)
        metric_rows.extend(current_metrics); bin_rows.extend(current_bins)
        print(f"evaluated {dataset}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    bins = pd.DataFrame(bin_rows)
    per_seed, summary = aggregate(metrics)
    comparisons = paired_comparisons(per_seed)
    native = evaluate_native_scores()
    native_summary = native.groupby(
        ["dataset", "method", "method_label"], as_index=False
    ).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"),
        spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
        top20_precision_mean=("top20_precision", "mean"),
        safe_prevalence=("safe_prevalence", "mean"),
    )
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS / "e3_contribution_identification_by_coalition.csv", index=False)
    per_seed.to_csv(RESULTS / "e3_contribution_identification_by_seed.csv", index=False)
    summary.to_csv(RESULTS / "e3_contribution_identification_summary.csv", index=False)
    comparisons.to_csv(RESULTS / "e3_paired_comparisons.csv", index=False)
    bins.to_csv(RESULTS / "e3_analytic_probability_deciles.csv", index=False)
    native.to_csv(RESULTS / "e3_native_score_by_seed.csv", index=False)
    native_summary.to_csv(RESULTS / "e3_native_score_summary.csv", index=False)
    metadata = {
        "datasets": list(CONFIGS), "seeds": list(SEEDS), "tau_nats": TAU,
        "coalition_aggregation": "macro average after per-coalition metrics",
        "test_labels_role": "evaluation only",
        "native_score_target": "benefit of removing one modality from the full alliance",
        "device": device,
    }
    (RESULTS / "e3_contribution_identification_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure(summary, bins, native,
                FIGURES / "e3_contribution_identification.png")
    print(summary.to_string(index=False), flush=True)
    print(native_summary.to_string(index=False).replace("I²", "I2"), flush=True)


if __name__ == "__main__":
    main()
