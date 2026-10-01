from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import sys
import time
from pathlib import Path

# Must be set before the first CUDA operation for reproducible cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from .io import load_json, write_json, sha256


def main():
    p = argparse.ArgumentParser(description="RCG fixed-feature observational study")
    p.add_argument("command", choices=["download", "prepare", "run", "analyze", "plot", "all",
                                               "coalition", "coalition-analyze", "coalition-all",
                                               "contribution"])
    p.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    p.add_argument("--data", default="data")
    p.add_argument("--config", default="config.json")
    p.add_argument("--run", help="output directory; defaults to runs/<dataset>")
    p.add_argument("--base-run", help="completed observational run supplying coalition-trained checkpoints")
    p.add_argument("--url", help="public feature mirror URL; published hash still mandatory")
    p.add_argument("--seeds", type=int, nargs="+", help="explicit smoke/pilot override, recorded in manifest")
    p.add_argument("--epochs", type=int)
    p.add_argument("--bootstrap", type=int)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--estimator-epochs", type=int, default=100)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    cfg = load_json(args.config)
    for key in ("seeds", "epochs", "bootstrap", "device"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if cfg["epochs"] < 1 or cfg["bootstrap"] < 1 or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        p.error("epochs/bootstrap must be positive and seeds unique")
    run = Path(args.run or f"runs/{args.dataset}")
    if args.command in ("download", "all"):
        from .download import download
        download(args.dataset, args.data, args.url)
    if args.command in ("prepare", "all"):
        from .data import prepare
        prepare(args.dataset, args.data, cfg["split_seed"])
    if args.command in ("run", "all"):
        import torch
        from .data import load_prepared
        from .train import train_seed, load_models
        from .evaluate import evaluate_seed
        torch.set_num_threads(cfg["threads"])
        device = "cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else cfg["device"]
        device = "cpu" if device == "auto" else device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        splits, audit = load_prepared(args.dataset, args.data)
        if audit["split_seed"] != cfg["split_seed"]:
            raise ValueError("Prepared split seed differs from config; run prepare again")
        run.mkdir(parents=True, exist_ok=True)
        # Refuse silently mixing changed protocols, datasets or code into a resumed run.
        source_hashes = {f.name: sha256(f) for f in Path(__file__).parent.glob("*.py")}
        manifest = {"dataset": args.dataset, "config": cfg, "data_audit": audit,
                    "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": device,
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                    "source_sha256": source_hashes}
        manifest_path = run / "manifest.json"
        if manifest_path.exists():
            old = load_json(manifest_path)
            if old["config"] != cfg or old["data_audit"] != audit or old["source_sha256"] != source_hashes:
                raise ValueError("Run manifest differs; choose a new --run directory to preserve prior results")
            if old["device"] != device or old["torch"] != torch.__version__:
                raise ValueError("Run environment changed; use the original device/environment or a new --run directory")
        else:
            write_json(manifest_path, manifest)
            snapshot = run / "source" / "rcg"
            snapshot.mkdir(parents=True, exist_ok=True)
            for source in Path(__file__).parent.glob("*.py"):
                shutil.copy2(source, snapshot / source.name)
        start = time.perf_counter()
        for seed in cfg["seeds"]:
            out = run / f"seed_{seed}"
            if (out / "evaluation_complete.json").exists():
                print(f"Seed {seed} already complete", flush=True)
                continue
            if (out / "checkpoint.pt").exists():
                models, temps = load_models(out / "checkpoint.pt", cfg, device)
            else:
                models, temps = train_seed(splits, cfg, seed, out, device)
            evaluate_seed(models, temps, splits, cfg, seed, out, device)
        write_json(run / "runtime.json", {"wall_seconds_this_invocation": time.perf_counter()-start})
    if args.command in ("analyze", "all"):
        from .analyze import analyze
        # Statistics use the run's immutable settings, except an explicit bootstrap override.
        actual = load_json(run / "manifest.json")["config"]
        if args.bootstrap is not None:
            actual["bootstrap"] = args.bootstrap
        analyze(run, actual)
    if args.command in ("plot", "all"):
        from .plot import plot
        plot(run)
    if args.command in ("coalition", "coalition-all"):
        import torch
        from .coalition import evaluate_coalitions
        from .data import load_prepared
        from .train import load_models
        torch.set_num_threads(cfg["threads"])
        device = "cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else cfg["device"]
        device = "cpu" if device == "auto" else device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        splits, audit = load_prepared(args.dataset, args.data)
        base = Path(args.base_run or f"runs/{args.dataset}")
        coalition_run = Path(args.run or f"runs/{args.dataset}-coalition")
        coalition_run.mkdir(parents=True, exist_ok=True)
        base_manifest = load_json(base / "manifest.json")
        manifest = {
            "dataset": args.dataset,
            "config": cfg,
            "data_audit": audit,
            "base_run": str(base.resolve()),
            "base_source_sha256": base_manifest.get("source_sha256"),
            "protocol": "all valid modality coalitions using base concat_masked; train prior for empty Shapley baseline",
        }
        manifest_path = coalition_run / "manifest.json"
        if manifest_path.exists() and load_json(manifest_path) != manifest:
            raise ValueError("Coalition manifest differs; choose a new --run directory")
        write_json(manifest_path, manifest)
        for seed in cfg["seeds"]:
            out = coalition_run / f"seed_{seed}"
            if (out / "evaluation_complete.json").exists():
                print(f"Coalition seed {seed} already complete", flush=True)
                continue
            models, temps = load_models(base / f"seed_{seed}" / "checkpoint.pt", cfg, device)
            evaluate_coalitions(models, temps, splits, cfg, args.dataset, seed, out, device)
    if args.command in ("coalition-analyze", "coalition-all"):
        from .coalition import analyze_coalitions
        coalition_run = Path(args.run or f"runs/{args.dataset}-coalition")
        actual = load_json(coalition_run / "manifest.json")["config"]
        if args.bootstrap is not None:
            actual["bootstrap"] = args.bootstrap
        analyze_coalitions(coalition_run, actual)
    if args.command == "contribution":
        import numpy as np
        import pandas as pd
        import torch
        from .contribution import (RelationalLossPredictor, crossfit_targets, evaluate_estimator,
                                   final_targets, summarize_estimator, train_estimator, tune_selector)
        from .data import load_prepared
        from .train import load_models, seed_all
        torch.set_num_threads(cfg["threads"])
        device = "cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else cfg["device"]
        device = "cpu" if device == "auto" else device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        splits, audit = load_prepared(args.dataset, args.data)
        base = Path(args.base_run or f"runs/{args.dataset}")
        contribution_run = Path(args.run or f"runs/{args.dataset}-contribution")
        contribution_run.mkdir(parents=True, exist_ok=True)
        prior_counts = np.bincount(splits["train"]["y"], minlength=2).astype(float)
        prior = prior_counts / prior_counts.sum()
        manifest = {
            "dataset": args.dataset, "config": cfg, "data_audit": audit,
            "base_run": str(base.resolve()), "folds": args.folds,
            "estimator_epochs": args.estimator_epochs,
            "protocol": "grouped OOF coalition-loss supervision; relational loss prediction; no test labels at inference",
        }
        manifest_path = contribution_run / "manifest.json"
        if manifest_path.exists() and load_json(manifest_path) != manifest:
            raise ValueError("Contribution manifest differs; choose a new --run directory")
        write_json(manifest_path, manifest)
        all_metrics = []
        dims = [splits["train"][m].shape[1] for m in ("text", "audio", "vision")]
        for seed in cfg["seeds"]:
            out = contribution_run / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / "evaluation_complete.json").exists():
                all_metrics.append(load_json(out / "metrics.json"))
                print(f"Contribution seed {seed} already complete", flush=True)
                continue
            target_path = out / "crossfit_targets.npz"
            if target_path.exists():
                with np.load(target_path) as saved:
                    train_targets = {key: saved[key] for key in saved.files}
                crossfit_meta = load_json(out / "crossfit_metadata.json")
            else:
                fold_cfg = dict(cfg)
                fold_cfg["epochs"] = min(cfg["epochs"], 50)
                fold_cfg["patience"] = min(cfg["patience"], 7)
                result = crossfit_targets(splits["train"], fold_cfg, seed, device, args.folds)
                crossfit_meta = result.pop("metadata")
                np.savez_compressed(target_path, **result)
                write_json(out / "crossfit_metadata.json", crossfit_meta)
                train_targets = result
            base_models, temperatures = load_models(base / f"seed_{seed}" / "checkpoint.pt", cfg, device)
            valid_targets = final_targets(base_models, temperatures, splits["calibration"], prior, device)
            test_targets = final_targets(base_models, temperatures, splits["test"], prior, device)
            seed_all(seed * 10000 + 777)
            estimator = RelationalLossPredictor(dims).to(device)
            history = train_estimator(estimator, splits["train"], train_targets,
                                      splits["calibration"], valid_targets, cfg, device,
                                      epochs=args.estimator_epochs, patience=10)
            policy, policy_grid = tune_selector(estimator, splits["calibration"], valid_targets, device)
            frame = evaluate_estimator(estimator, splits["test"], test_targets, seed, device,
                                       policy["risk_lambda"], policy["margin"])
            frame.to_parquet(out / "predictions.parquet", index=False)
            metrics = {"dataset": args.dataset, "train_seed": seed, **summarize_estimator(frame)}
            write_json(out / "metrics.json", metrics)
            write_json(out / "training.json", {"history": history, "crossfit": crossfit_meta,
                                                "selector_policy": policy, "selector_grid": policy_grid})
            torch.save({"state_dict": estimator.state_dict(), "dims": dims}, out / "estimator.pt")
            write_json(out / "evaluation_complete.json", {"seed": seed, "n_test": len(frame)})
            all_metrics.append(metrics)
            print(f"Contribution seed {seed}: regret reduction={metrics['regret_reduction_fraction']:.3f}", flush=True)
            del base_models, estimator
            if device == "cuda":
                torch.cuda.empty_cache()
        metric_frame = pd.DataFrame(all_metrics)
        metric_frame.to_csv(contribution_run / "metrics_by_seed.csv", index=False)
        numeric = [c for c in metric_frame.columns if c not in ("dataset", "train_seed")]
        summary = pd.DataFrame({"metric": numeric,
                                "mean": [metric_frame[c].mean() for c in numeric],
                                "std": [metric_frame[c].std(ddof=1) for c in numeric]})
        summary.to_csv(contribution_run / "metrics_summary.csv", index=False)
        lines = [f"# {args.dataset.upper()} 关系贡献预测器结果", "",
                 "五折按原视频交叉拟合联盟损失监督；测试推理不使用标签或真实联盟损失。", "",
                 "| metric | mean | std |", "|---|---:|---:|"]
        lines.extend(f"| {row.metric} | {row['mean']:.4f} | {row['std']:.4f} |" for _, row in summary.iterrows())
        (contribution_run / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
