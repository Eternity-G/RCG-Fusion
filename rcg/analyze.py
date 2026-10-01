from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import MODALITIES
from .evaluate import METHODS
from .io import load_json, write_json, sha256
from .stats import diagnostics, task_metrics, bootstrap_ratios, centered_bootstrap_p, holm

GROUP = ["condition", "family", "kind", "affected", "severity", "modality"]
METRICS = ["spearman", "hcr", "high_mean_harm", "negative_flip_rate", "high_negative_flip_rate",
           "harmful_auroc", "harmful_auprc", "harmful_prevalence", "high_coverage"]


def analyze(run, cfg):
    run = Path(run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    seed_dirs = [run / f"seed_{s}" for s in cfg["seeds"]]
    for d in seed_dirs:
        if not (d / "evaluation_complete.json").exists():
            raise ValueError(f"Incomplete seed {d.name}; finish evaluation before analysis")
    all_seed_rows, task_rows, context_rows, bin_rows, cluster_rows, trajectory_rows = [], [], [], [], [], []
    unique_high_sets, unique_sets = {}, {}
    for method in METHODS:
        for d in seed_dirs:
            seed = int(d.name.split("_")[1])
            f = pd.read_parquet(d / f"{method}.parquet")
            main = f[f.family == "main"]
            clean = main[main.kind == "clean"]
            for (_, kind, affected, severity), group in main.groupby(["condition", "kind", "affected", "severity"], sort=False):
                # One task prediction per sample/corruption realization, not per removed modality.
                g = group.drop_duplicates(["sample_id", "corruption_seed"])
                task_rows.append({"method": method, "seed": seed, "kind": kind, "affected": affected,
                                  "severity": severity, **task_metrics(g.y.to_numpy(), g[["full_p0", "full_p1"]].to_numpy())})
            if method == "uniform":
                for m in MODALITIES:
                    g = clean[clean.modality == m]
                    for variant, column in (("raw", "mono_raw_p1"), ("calibrated", "mono_cal_p1")):
                        p = g[column].to_numpy()
                        task_rows.append({"method": f"uni_{m}_{variant}", "seed": seed, "kind": "clean", "affected": "none",
                                          "severity": 0.0, **task_metrics(g.y.to_numpy(), np.column_stack([1-p, p]))})
            if method == "tmc":
                for m in MODALITIES:
                    g = clean[clean.modality == m]
                    alpha = g[["tmc_alpha0", "tmc_alpha1"]].to_numpy()
                    p = alpha/alpha.sum(1, keepdims=True)
                    task_rows.append({"method": f"tmc_uni_{m}", "seed": seed, "kind": "clean", "affected": "none",
                                      "severity": 0.0, **task_metrics(g.y.to_numpy(), p)})
            variants = [("primary", "r", "high")]
            if method in ("concat_masked", "concat_clean"):
                variants.append(("calibrated_probe", "r_calibrated", "high_calibrated"))
            for variant, score, high in variants:
                for keys, g in main.groupby(GROUP, sort=False):
                    metadata = dict(zip(GROUP, keys))
                    for epsilon in cfg["epsilons"]:
                        all_seed_rows.append({"method": method, "score": variant, "seed": seed,
                                              "epsilon": epsilon, **metadata, **diagnostics(g, epsilon, score, high)})
                    # Cluster CIs for the preregistered primary epsilon; sensitivity rows have seed SD.
                    key = (method, variant, metadata["condition"], metadata["modality"])
                    h = g[high].to_numpy(dtype=bool)
                    unique_high_sets.setdefault(key, set()).update(g.loc[h, "sample_id"])
                    unique_sets.setdefault(key, set()).update(g.sample_id)
                    contribution = pd.DataFrame({"video_id": g.video_id,
                        "high_n": h.astype(float), "harm_n": h*(g.u.to_numpy() < -0.01),
                        "harm_sum": h*np.maximum(-g.u.to_numpy(), 0),
                        "flip_n": g.negative_flip.to_numpy(dtype=float), "n": np.ones(len(g))})
                    for video, row in contribution.groupby("video_id").sum().iterrows():
                        cluster_rows.append({"method": method, "score": variant, **metadata,
                                             "video_id": video, "seed": seed, **row.to_dict()})
                for m, g in clean.groupby("modality"):
                    bins = np.minimum((g[score].to_numpy()*10).astype(int), 9)
                    for b in np.unique(bins):
                        bg = g[bins == b]
                        bin_rows.append({"method": method, "score": variant, "seed": seed, "modality": m,
                                         "bin": int(b), "r_mean": bg[score].mean(), "u_mean": bg.u.mean(), "n": len(bg)})
            # Context compares to the very same target sample at clean, unchanged target input.
            context = f[f.family == "context"].copy()
            base = clean[["sample_id", "modality", "r", "u"]].rename(columns={"r": "r_clean", "u": "u_clean"})
            context = context.merge(base, on=["sample_id", "modality"], validate="many_to_one")
            if not np.allclose(context.r, context.r_clean, atol=1e-6):
                raise AssertionError("Intrinsic target reliability changed despite frozen target input")
            context["switch"] = ((context.u > .01) & (context.u_clean < -.01)) | ((context.u < -.01) & (context.u_clean > .01))
            for (m, severity), g in context.groupby(["modality", "severity"]):
                context_rows.append({"method": method, "seed": seed, "modality": m, "severity": severity,
                                     "switch_rate": g.switch.mean(), "mean_abs_delta_u": (g.u-g.u_clean).abs().mean(),
                                     "n_unique": g.sample_id.nunique()})
            # All target trajectories retained as source data; plotting selects deterministic examples.
            trajectory_rows.append(context[["sample_id", "video_id", "seed", "method", "modality", "severity",
                                            "corruption_seed", "r", "u", "u_clean", "switch"]])
            del f, main, context
        print(f"Analyzed predictions: {method}", flush=True)
    rows = pd.DataFrame(all_seed_rows)
    rows.to_csv(out / "diagnostics_by_seed.csv", index=False)
    task_table = pd.DataFrame(task_rows)
    task_table.to_csv(out / "task_metrics_by_seed.csv", index=False)
    task_summary = task_table.groupby(["method", "kind", "affected", "severity"])[
        ["accuracy", "macro_f1", "nll", "brier", "ece"]].agg(["mean", "std"])
    task_summary.columns = ["_".join(c) for c in task_summary.columns]
    task_summary.reset_index().to_csv(out / "task_metrics_summary.csv", index=False)
    context_table = pd.DataFrame(context_rows)
    context_table.to_csv(out / "context_by_seed.csv", index=False)
    context_summary = context_table.groupby(["method", "modality", "severity"])[
        ["switch_rate", "mean_abs_delta_u"]].agg(["mean", "std"])
    context_summary.columns = ["_".join(c) for c in context_summary.columns]
    context_summary.reset_index().to_csv(out / "context_summary.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(out / "reliability_bins_by_seed.csv", index=False)
    pd.concat(trajectory_rows, ignore_index=True).to_parquet(out / "context_trajectories.parquet", index=False)
    group_keys = ["method", "score", "epsilon"] + GROUP
    summary = rows.groupby(group_keys, dropna=False)[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, c)).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    cluster = pd.DataFrame(cluster_rows)
    cluster.to_parquet(out / "cluster_sufficient_statistics.parquet", index=False)
    inference_keys = ["method", "score"] + GROUP
    sums = cluster.groupby(inference_keys+["video_id"], dropna=False)[["high_n", "harm_n", "harm_sum", "flip_n", "n"]].sum().reset_index()
    videos = sorted(sums.video_id.unique())
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    weights = rng.multinomial(len(videos), np.full(len(videos), 1/len(videos)), size=cfg["bootstrap"]).astype(float)
    infer_rows, distributions = [], {}
    for keys, g in sums.groupby(inference_keys, dropna=False, sort=False):
        meta = dict(zip(inference_keys, keys))
        key = (meta["method"], meta["score"], meta["condition"], meta["modality"])
        g = g.set_index("video_id").reindex(videos, fill_value=0)
        numerator = g[["harm_n", "harm_sum", "flip_n"]].to_numpy()
        denominator = g[["high_n", "high_n", "n"]].to_numpy()
        estimate = np.divide(numerator.sum(0), denominator.sum(0), out=np.full(3, np.nan), where=denominator.sum(0)>0)
        draws = bootstrap_ratios(numerator, denominator, weights)
        high_count = len(unique_high_sets[key])
        minimum_seed_high = int(rows[(rows.method == meta["method"]) & (rows.score == meta["score"]) &
                                    (rows.condition == meta["condition"]) & (rows.modality == meta["modality"]) &
                                    np.isclose(rows.epsilon, .01)].n_high_unique.min())
        descriptive = minimum_seed_high < cfg["min_unique_high"]
        row = {**meta, "epsilon": 0.01, "n_high_unique_union_seeds": high_count,
               "n_high_min_seed": minimum_seed_high,
               "n_unique": len(unique_sets[key]), "n_videos": len(videos), "descriptive_only": descriptive}
        for i, name in enumerate(("hcr", "high_mean_harm", "negative_flip_rate")):
            valid = draws[:, i][np.isfinite(draws[:, i])]
            ci = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
            row[f"{name}_pooled"] = estimate[i]
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = ci
        infer_rows.append(row)
        distributions[key] = (estimate, draws)
    inference = pd.DataFrame(infer_rows)
    inference.to_csv(out / "cluster_intervals.csv", index=False)
    summary.to_csv(out / "diagnostics_summary.csv", index=False)
    # One prespecified family: calibrated minus uncalibrated HCR across main conditions/modalities.
    comparisons = []
    for key, (estimate_b, draws_b) in distributions.items():
        method, score, cond, modality = key
        if method != "confidence_calibrated":
            continue
        other = ("confidence", "primary", cond, modality)
        estimate_a, draws_a = distributions[other]
        delta, samples = estimate_b[0]-estimate_a[0], draws_b[:, 0]-draws_a[:, 0]
        condition_rows = inference[(inference.condition == cond) & (inference.modality == modality) &
                                   (inference.method.isin(["confidence", "confidence_calibrated"])) &
                                   (inference.score == "primary")]
        enough = not condition_rows.descriptive_only.any()
        finite = samples[np.isfinite(samples)]
        interval = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
        comparisons.append({"condition": cond, "modality": modality, "comparison": "calibrated - raw HCR",
                            "delta": delta, "ci_low": interval[0], "ci_high": interval[1],
                            "descriptive_only": not enough,
                            "p_bootstrap": centered_bootstrap_p(samples, delta) if enough else np.nan})
    compare = pd.DataFrame(comparisons)
    compare["p_holm"] = holm(compare.p_bootstrap)
    compare.to_csv(out / "paired_calibration_comparisons.csv", index=False)
    # Second, separate prespecified family: within-method stress minus clean HCR.
    stress_rows = []
    for key, (estimate, draws) in distributions.items():
        method, score, cond, modality = key
        if cond == "main/clean/none/0":
            continue
        base_key = (method, score, "main/clean/none/0", modality)
        base, base_draws = distributions[base_key]
        delta, samples = estimate[0]-base[0], draws[:, 0]-base_draws[:, 0]
        finite = samples[np.isfinite(samples)]
        interval = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
        support = inference[(inference.method == method) & (inference.score == score) &
                            (inference.modality == modality) & (inference.condition.isin([cond, base_key[2]]))]
        enough = not support.descriptive_only.any()
        stress_rows.append({"method": method, "score": score, "condition": cond, "modality": modality,
                           "comparison": "stress - clean HCR", "delta": delta, "ci_low": interval[0], "ci_high": interval[1],
                           "descriptive_only": not enough,
                           "p_bootstrap": centered_bootstrap_p(samples, delta) if enough else np.nan})
    stress = pd.DataFrame(stress_rows)
    stress["p_holm"] = holm(stress.p_bootstrap)
    stress.to_csv(out / "paired_stress_comparisons.csv", index=False)
    write_json(out / "analysis_metadata.json", {
        "seeds": cfg["seeds"], "bootstrap": cfg["bootstrap"], "cluster": "original video",
        "ci_estimand": "pooled conditional ratio across fixed training seeds and corruption repeats; resample videos only",
        "seed_variability": "separate arithmetic seed mean and sample SD; not bootstrap independence",
        "pvalue": "approximate null-centered paired cluster-bootstrap, two-sided; Holm across all main HCR calibration comparisons",
        "low_count": "intervals retained descriptively; no decision or p-value below minimum unique high-confidence count",
        "analysis_source_sha256": {name: sha256(Path(__file__).parent / name)
                                   for name in ("analyze.py", "stats.py")},
    })
    report(run, inference, pd.DataFrame(task_rows), cfg)


def report(run, inference, task, cfg):
    lines = ["# RCG 观测实验结果", "", "这些结果描述固定表示、固定融合模型及指定删除协议；不等于现实因果贡献。", "",
             f"训练种子：{cfg['seeds']}；原视频簇 bootstrap：{cfg['bootstrap']} 次。", "",
             "## 干净测试集任务表现", "", "| 方法 | Accuracy 均值 | Macro-F1 均值 | NLL 均值 |", "|---|---:|---:|---:|"]
    clean = task[task.kind == "clean"]
    for name, g in clean.groupby("method"):
        lines.append(f"| {name} | {g.accuracy.mean():.4f} | {g.macro_f1.mean():.4f} | {g.nll.mean():.4f} |")
    lines += ["", "## 校准后与证据融合的主要证据", "", "| 方法/分数 | 模态 | 干净 HCR | 95% 视频簇 CI | 每个种子的最少高置信样本 | 高置信平均损害 | 判读 |", "|---|---|---:|---|---:|---:|---|"]
    focus = inference[(inference.kind == "clean") & ((inference.method == "confidence_calibrated") |
                       (inference.method == "tmc") |
                       ((inference.method == "concat_masked") & (inference.score == "calibrated_probe")))]
    for _, row in focus.iterrows():
        status = "仅描述：样本不足" if row.descriptive_only else ("HCR 下界超过项目 5% 门槛" if row.hcr_ci_low > .05 else "未达到项目门槛")
        lines.append(f"| {row.method}/{row.score} | {row.modality} | {row.hcr_pooled:.3f} | [{row.hcr_ci_low:.3f}, {row.hcr_ci_high:.3f}] | {row.n_high_min_seed} | {row.high_mean_harm_pooled:.3f} | {status} |")
    candidate_modalities = []
    for modality in MODALITIES:
        candidate = focus[(focus.modality == modality) &
                          (((focus.method == "confidence_calibrated") & (focus.score == "primary")) |
                           ((focus.method == "concat_masked") & (focus.score == "calibrated_probe")))]
        if len(candidate) == 2 and (~candidate.descriptive_only).all() and (candidate.hcr_ci_low > .05).all() and (candidate.high_mean_harm_pooled > 0).all():
            candidate_modalities.append(modality)
    lines += ["", "## 预注册项目门槛判读", ""]
    if candidate_modalities:
        lines.append("- 同时通过概率融合和带缺失训练的学习式融合分项门槛的模态：" + "、".join(candidate_modalities) + "。")
    else:
        lines.append("- 没有模态同时通过概率融合和带缺失训练的学习式融合分项门槛。")
    lines.append("- 这只是继续研究的工程门槛；最终判断还需跨数据集重复、代表性可靠融合复验和机制分析。")
    lines += ["", "## 解释限制", "", "- 5% 是项目继续投入门槛，不是领域公认证明标准。需同时检查损害幅度、负向翻转和跨方法复验。",
              "- HCR 的校准前后分母是各自阈值选出的样本；配对的是原视频，不能解释为同一高置信子群上的处理效应。",
              "- 温度在单模态校准集上拟合；它可能改变融合权重和高置信子集。校准后 HCR 上升不等于校准使模型整体更差。",
              "- concat_clean 为缺失输入伪象对照，不用于独立建立结论。concat 的分数来自单模态探针，不是模型原生融合权重。",
              "- TMC 是相同池化特征上的适配（Softplus 证据头），不是原论文数值复现。",
              "- 主指标为损失贡献；负损失贡献不一定引起分类翻转。强扰动不代表现实噪声。",
              "- bootstrap 固定已训练种子，不将种子、片段或噪声重复视为独立视频；跨训练种子波动另报。",
              "- 不足 30 个独立高置信样本的条件仅描述；相关性较低不能单独证明分数无用。", "",
              "全部条件、容差敏感性、任务校准、上下文变号率和校准配对检验见 analysis/ 下 CSV。"]
    (Path(run) / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
