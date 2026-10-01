from __future__ import annotations

import math
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import MODALITIES
from .evaluate import conditions, corrupt, branches, reliability
from .io import load_json, write_json
from .train import tensors


SHORT = {"text": "T", "audio": "A", "vision": "V"}


def coalition_masks(n_modalities=3, include_empty=True):
    """Return masks in deterministic size/lexicographic order."""
    masks = []
    if include_empty:
        masks.append((0,) * n_modalities)
    for size in range(1, n_modalities + 1):
        for members in combinations(range(n_modalities), size):
            mask = tuple(int(i in members) for i in range(n_modalities))
            masks.append(mask)
    return tuple(masks)


MASKS = coalition_masks()


def coalition_name(mask):
    name = "".join(SHORT[m] for m, keep in zip(MODALITIES, mask) if keep)
    return name or "EMPTY"


def exact_shapley(losses, available):
    """Exact per-sample Shapley values for value=-loss over available modalities.

    losses maps bit-mask tuples to [N] loss arrays and must contain the empty
    coalition plus every subset of ``available``. Positive values mean that a
    modality reduces loss on average over all predecessor coalitions.
    """
    available = tuple(int(x) for x in available)
    players = [i for i, keep in enumerate(available) if keep]
    n_players = len(players)
    if not n_players:
        raise ValueError("At least one modality must be available")
    n_samples = len(next(iter(losses.values())))
    output = np.full((n_samples, len(available)), np.nan, dtype=np.float64)
    denominator = math.factorial(n_players)
    for m in players:
        phi = np.zeros(n_samples, dtype=np.float64)
        others = [i for i in players if i != m]
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n_players-size-1) / denominator
            for members in combinations(others, size):
                before = tuple(int(i in members) for i in range(len(available)))
                after = list(before)
                after[m] = 1
                phi += weight * (losses[before] - losses[tuple(after)])
        output[:, m] = phi
    return output


def coalition_statistics(losses, available):
    """Return Shapley, full-context deletion utility, regret and oracle subset."""
    available = tuple(int(x) for x in available)
    valid = [m for m in MASKS if any(m) and all(not keep or available[i] for i, keep in enumerate(m))]
    full = available
    if full not in losses:
        raise KeyError(f"Missing full available coalition {full}")
    shapley = exact_shapley(losses, available)
    deletion = np.full_like(shapley, np.nan)
    for i, keep in enumerate(available):
        if keep:
            without = list(full)
            without[i] = 0
            deletion[:, i] = losses[tuple(without)] - losses[full]
    matrix = np.stack([losses[m] for m in valid], axis=1)
    best = matrix.argmin(1)
    oracle = np.array([coalition_name(valid[i]) for i in best])
    regret = losses[full] - matrix[np.arange(len(matrix)), best]
    return shapley, deletion, regret, oracle


@torch.no_grad()
def evaluate_coalitions(models, temps, splits, cfg, dataset, seed, output, device):
    """Evaluate every valid coalition for every registered test condition."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    test = splits["test"]
    y = test["y"]
    counts = np.bincount(splits["train"]["y"], minlength=2).astype(np.float64)
    prior = counts / counts.sum()
    empty_loss = -np.log(np.maximum(prior[y], 1e-12))
    writer = None
    try:
        for ci, condition in enumerate(conditions(cfg)):
            arrays, present_np = corrupt([test[m] for m in MODALITIES], condition)
            available = tuple(int(x) for x in present_np[0])
            if not np.all(present_np == present_np[0]):
                raise ValueError("Coalition evaluator expects condition-level availability")
            xs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in arrays]
            raw, calibrated, _ = branches(models, temps, xs)
            raw_r = reliability(raw)[0].cpu().numpy()
            cal_r = reliability(calibrated)[0].cpu().numpy()
            losses = {(0, 0, 0): empty_loss}
            probabilities = {(0, 0, 0): np.repeat(prior[None, :], len(y), axis=0)}
            for mask in MASKS[1:]:
                if any(keep and not available[i] for i, keep in enumerate(mask)):
                    continue
                present = torch.tensor(mask, dtype=torch.float32, device=device)[None].repeat(len(y), 1)
                # Persist a stable float64 schema even when a later missing-modality
                # condition fills an invalid coalition with NaN.
                prob = models["concat_masked"](xs, present).softmax(-1).cpu().numpy().astype(np.float64)
                probabilities[mask] = prob
                losses[mask] = -np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12))
            shapley, deletion, regret, oracle = coalition_statistics(losses, available)
            full = probabilities[available]
            frame = pd.DataFrame({
                "dataset": dataset,
                "sample_id": test["id"],
                "group_or_video_id": test["video"],
                "train_seed": seed,
                **asdict(condition),
                "condition": condition.key,
                "available_modalities": coalition_name(available),
                "label": y.astype(np.int8),
                "full_p0": full[:, 0],
                "full_p1": full[:, 1],
                "full_loss": losses[available],
                "fusion_regret": regret,
                "oracle_coalition": oracle,
            })
            for mi, modality in enumerate(MODALITIES):
                frame[f"reliability_raw_{modality}"] = raw_r[:, mi]
                frame[f"reliability_calibrated_{modality}"] = cal_r[:, mi]
                frame[f"conditional_contribution_{modality}"] = deletion[:, mi]
                frame[f"shapley_contribution_{modality}"] = shapley[:, mi]
            for mask in MASKS:
                name = coalition_name(mask)
                if mask in losses:
                    frame[f"loss_{name}"] = losses[mask]
                    frame[f"p1_{name}"] = probabilities[mask][:, 1]
                else:
                    frame[f"loss_{name}"] = np.nan
                    frame[f"p1_{name}"] = np.nan
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output / "coalitions.parquet", table.schema, compression="zstd")
            writer.write_table(table)
            if ci % 20 == 0:
                print(f"seed={seed}: coalition condition {ci+1}, {condition.key}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    write_json(output / "evaluation_complete.json", {
        "dataset": dataset,
        "seed": seed,
        "conditions": ci + 1,
        "coalitions": [coalition_name(x) for x in MASKS],
        "empty_coalition": "training-label prior; used only as the Shapley baseline",
        "model": "concat_masked from the immutable base observational run",
    })


def _mean_ci(values, groups, bootstrap, seed):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    sums = np.array([values[groups == g].sum() for g in unique])
    counts = np.array([(groups == g).sum() for g in unique])
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap, dtype=float)
    for start in range(0, bootstrap, 500):
        size = min(500, bootstrap-start)
        idx = rng.integers(0, len(unique), size=(size, len(unique)))
        draws[start:start+size] = sums[idx].sum(1) / counts[idx].sum(1)
    return float(values.mean()), *np.quantile(draws, [.025, .975]).tolist(), len(unique)


def _ratio_ci(numerator, denominator, groups, bootstrap, seed):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    nums = np.array([numerator[groups == g].sum() for g in unique])
    dens = np.array([denominator[groups == g].sum() for g in unique])
    estimate = numerator.sum() / denominator.sum() if denominator.sum() else np.nan
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, bootstrap, 500):
        size = min(500, bootstrap-start)
        idx = rng.integers(0, len(unique), size=(size, len(unique)))
        d = dens[idx].sum(1)
        n = nums[idx].sum(1)
        draws.extend((n[d > 0] / d[d > 0]).tolist())
    low, high = np.quantile(draws, [.025, .975]) if draws else (np.nan, np.nan)
    return float(estimate), float(low), float(high), len(unique), int(denominator.sum())


def _markdown_table(frame):
    """Small dependency-free Markdown renderer for generated reports."""
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    headers = [str(x) for x in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def analyze_coalitions(run, cfg):
    run = Path(run)
    manifest = load_json(run / "manifest.json") if (run / "manifest.json").exists() else None
    base_run = Path(manifest["base_run"]) if manifest and "base_run" in manifest else None
    paths = sorted(run.glob("seed_*/coalitions.parquet"))
    if not paths:
        raise FileNotFoundError(f"No coalition outputs under {run}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    clean = frame[(frame.family == "main") & (frame.kind == "clean")].copy()
    analysis = run / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    rows = []
    bootstrap = int(cfg["bootstrap"])
    base_seed = int(cfg["bootstrap_seed"])
    estimate, low, high, n_groups = _mean_ci(
        clean.fusion_regret, clean.group_or_video_id, bootstrap, base_seed
    )
    rows.append({"metric": "mean_fusion_regret", "modality": "all", "estimate": estimate,
                 "ci_low": low, "ci_high": high, "n_groups": n_groups,
                 "n_high": np.nan, "contribution": "oracle_subset",
                 "threshold_source": "not_applicable"})
    for mi, modality in enumerate(MODALITIES):
        reliability_col = f"reliability_calibrated_{modality}"
        if base_run is not None:
            threshold_map = {
                int(seed): float(load_json(base_run / f"seed_{seed}" / "thresholds.json")["calibrated"][mi])
                for seed in clean.train_seed.unique()
            }
            threshold_by_row = clean.train_seed.map(threshold_map).to_numpy()
            threshold_source = "clean_calibration_q90"
        else:
            # Synthetic software fixtures do not have an immutable base run.
            threshold_by_row = clean.groupby("train_seed")[reliability_col].transform(
                lambda x: x.quantile(cfg["quantile"])
            ).to_numpy()
            threshold_source = "test_q90_synthetic_fixture_only"
        high_mask = clean[reliability_col].to_numpy() >= threshold_by_row
        for contribution_name, prefix in (("deletion", "conditional_contribution"), ("shapley", "shapley_contribution")):
            utility = clean[f"{prefix}_{modality}"].to_numpy()
            valid = np.isfinite(utility)
            denominator = high_mask & valid
            numerator = denominator & (utility < -0.01)
            estimate, low, high, n_groups, n_high = _ratio_ci(
                numerator, denominator, clean.group_or_video_id, bootstrap, base_seed + 10 + mi
            )
            rows.append({"metric": "high_reliability_harm_rate", "modality": modality,
                         "estimate": estimate, "ci_low": low, "ci_high": high,
                         "n_groups": n_groups, "n_high": n_high, "contribution": contribution_name,
                         "threshold_source": threshold_source})
    intervals = pd.DataFrame(rows)
    intervals.to_csv(analysis / "clean_cluster_intervals.csv", index=False)

    by_seed = []
    for seed, part in clean.groupby("train_seed"):
        record = {
            "train_seed": seed,
            "n": len(part),
            "mean_fusion_regret": part.fusion_regret.mean(),
            "regret_gt_0.01": (part.fusion_regret > .01).mean(),
            "oracle_full_rate": (part.oracle_coalition == part.available_modalities).mean(),
            "oracle_single_rate": part.oracle_coalition.str.len().eq(1).mean(),
            "oracle_pair_rate": part.oracle_coalition.str.len().eq(2).mean(),
        }
        for modality in MODALITIES:
            record[f"mean_abs_shapley_minus_deletion_{modality}"] = np.nanmean(np.abs(
                part[f"shapley_contribution_{modality}"] - part[f"conditional_contribution_{modality}"]
            ))
        by_seed.append(record)
    pd.DataFrame(by_seed).to_csv(analysis / "clean_by_seed.csv", index=False)

    # Target reliability is unchanged by construction in context conditions. Join
    # clean and stressed rows to measure contribution sign changes directly.
    context_rows = []
    context = frame[(frame.family == "context") & (frame.kind == "gaussian")]
    key = ["sample_id", "train_seed"]
    clean_cols = key + [f"conditional_contribution_{m}" for m in MODALITIES]
    baseline = clean[clean_cols].drop_duplicates(key)
    for (target, severity), part in context.groupby(["affected", "severity"]):
        joined = part.merge(baseline, on=key, suffixes=("_stress", "_clean"), validate="many_to_one")
        before = joined[f"conditional_contribution_{target}_clean"]
        after = joined[f"conditional_contribution_{target}_stress"]
        score = f"reliability_calibrated_{target}"
        clean_score = clean[key + [score]].drop_duplicates(key)
        joined = joined.merge(clean_score, on=key, suffixes=("", "_clean_score"), validate="many_to_one")
        invariant = np.max(np.abs(joined[score] - joined[f"{score}_clean_score"]))
        context_rows.append({
            "target": target,
            "severity": severity,
            "n": len(joined),
            "positive_to_negative": ((before > .01) & (after < -.01)).mean(),
            "negative_to_positive": ((before < -.01) & (after > .01)).mean(),
            "bidirectional_switch_rate": (((before > .01) & (after < -.01)) | ((before < -.01) & (after > .01))).mean(),
            "max_target_reliability_change": invariant,
        })
    context_summary = pd.DataFrame(context_rows)
    context_summary.to_csv(analysis / "context_switches.csv", index=False)

    dataset = str(frame.dataset.iloc[0])
    go_hcr = intervals[(intervals.metric == "high_reliability_harm_rate") &
                       (intervals.contribution == "shapley") & (intervals.ci_low > .05)]
    sigma_one = context_summary[np.isclose(context_summary.severity, 1.0)]
    report = [
        f"# {dataset.upper()} 联盟一致贡献观测结果",
        "",
        "## 协议",
        "",
        "统一使用已完成缺失训练的 `concat_masked` 模型评估七个非空联盟；空联盟是训练集类别先验，仅作为精确 Shapley 基准。贡献与后悔值均为样本级交叉熵差。",
        "",
        "## 干净测试集",
        "",
        f"- 平均完整融合后悔值：{rows[0]['estimate']:.4f} nats（视频簇 95% CI {rows[0]['ci_low']:.4f}–{rows[0]['ci_high']:.4f}）。",
        f"- Shapley 高可靠有害率 CI 下界超过 5% 的模态：{', '.join(go_hcr.modality) if len(go_hcr) else '无'}。",
        "",
        _markdown_table(intervals),
        "",
        "## 其他模态条件变化",
        "",
        _markdown_table(sigma_one),
        "",
        "## 限制",
        "",
        "这些结果仍来自同一冻结表示与同一任务族；oracle 联盟使用测试标签，仅用于评估上界，不能部署。Shapley 是当前模型和损失下的归因，不是模态的现实因果价值。",
    ]
    (run / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(analysis / "analysis_metadata.json", {
        "rows": len(frame), "seeds": sorted(frame.train_seed.unique().tolist()),
        "bootstrap": bootstrap, "epsilon": .01, "quantile": cfg["quantile"],
    })
    return intervals
