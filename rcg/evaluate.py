from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import MODALITIES
from .io import write_json
from .models import probability_fusion, ds_fusion
from .train import tensors

METHODS = ("uniform", "confidence", "confidence_calibrated", "concat_masked", "concat_clean", "tmc")


@dataclass(frozen=True)
class Condition:
    family: str
    kind: str
    affected: str
    severity: float
    corruption_seed: int

    @property
    def key(self):
        return f"{self.family}/{self.kind}/{self.affected}/{self.severity:g}"


def conditions(cfg):
    yield Condition("main", "clean", "none", 0.0, 0)
    for m in MODALITIES:
        for kind, levels in (("gaussian", cfg["noise_levels"]), ("mask", cfg["mask_levels"])):
            for level in levels:
                for seed in cfg["corruption_seeds"]:
                    yield Condition("main", kind, m, float(level), seed)
        yield Condition("main", "missing", m, 1.0, 0)
    for target in MODALITIES:
        for level in cfg["noise_levels"]:
            for seed in cfg["corruption_seeds"]:
                yield Condition("context", "gaussian", target, float(level), seed)


def corrupt(xs, condition):
    result = [x.copy() for x in xs]
    present = np.ones((len(xs[0]), 3), dtype=np.float32)
    if condition.kind == "clean":
        return result, present
    for i, m in enumerate(MODALITIES):
        selected = (m != condition.affected) if condition.family == "context" else (m == condition.affected)
        if not selected:
            continue
        # Same random field across methods, model seeds and severity levels.
        rng = np.random.default_rng(np.random.SeedSequence([condition.corruption_seed, i, 811]))
        if condition.kind == "gaussian":
            if condition.severity:
                result[i] += condition.severity*rng.standard_normal(result[i].shape).astype(np.float32)
        elif condition.kind == "mask":
            result[i][rng.random(result[i].shape) < condition.severity] = 0
        elif condition.kind == "missing":
            result[i][:] = 0
            present[:, i] = 0
    return result, present


def reliability(prob):
    confidence = prob.max(-1).values
    entropy = 1+(prob.clamp_min(1e-12)*prob.clamp_min(1e-12).log()).sum(-1)/np.log(2)
    return confidence, entropy


@torch.no_grad()
def branches(models, temps, xs):
    logits = torch.stack([models[f"uni_{m}"](x) for m, x in zip(MODALITIES, xs)], 1)
    t = torch.tensor([temps[m] for m in MODALITIES], device=logits.device)[None, :, None]
    return logits.softmax(-1), (logits/t).softmax(-1), models["tmc"](xs)


@torch.no_grad()
def predict(method, models, xs, present, raw, calibrated, alphas):
    if method == "uniform":
        return probability_fusion(raw, present)
    if method in ("confidence", "confidence_calibrated"):
        return probability_fusion(calibrated if method.endswith("calibrated") else raw, present, True)
    if method.startswith("concat"):
        return models[method](xs, present).softmax(-1)
    a = ds_fusion(alphas, present)
    return a/a.sum(1, keepdim=True)


@torch.no_grad()
def evaluate_seed(models, temps, splits, cfg, seed, output, device):
    output = Path(output)
    cx, _ = tensors(splits["calibration"], device)
    raw, cal, alphas = branches(models, temps, cx)
    thresholds = {
        "raw": np.quantile(reliability(raw)[0].cpu().numpy(), cfg["quantile"], axis=0),
        "calibrated": np.quantile(reliability(cal)[0].cpu().numpy(), cfg["quantile"], axis=0),
        "native": np.quantile((1-2/alphas.sum(-1)).cpu().numpy(), cfg["quantile"], axis=0),
    }
    write_json(output / "thresholds.json", thresholds)
    test = splits["test"]
    y = test["y"]
    writers = {}
    # Only numeric/string columns; parquet compression avoids enormous repeated CSV files.
    try:
        for ci, condition in enumerate(conditions(cfg)):
            arrays, present_np = corrupt([test[m] for m in MODALITIES], condition)
            xs = [torch.as_tensor(x, device=device) for x in arrays]
            present = torch.as_tensor(present_np, device=device)
            raw, cal, alphas = branches(models, temps, xs)
            raw_r, raw_h = [a.cpu().numpy() for a in reliability(raw)]
            cal_r, _ = [a.cpu().numpy() for a in reliability(cal)]
            native_r = (1-2/alphas.sum(-1)).cpu().numpy()
            native_alpha = alphas.cpu().numpy()
            mono_raw = raw.cpu().numpy()
            mono_cal = cal.cpu().numpy()
            for method in METHODS:
                full = predict(method, models, xs, present, raw, cal, alphas).cpu().numpy()
                if not np.isfinite(full).all() or not np.allclose(full.sum(1), 1, atol=1e-5):
                    raise FloatingPointError(f"Invalid predictive probabilities: {method}/{condition.key}")
                loss_full = -np.log(np.maximum(full[np.arange(len(y)), y], 1e-12))
                frames = []
                for mi, m in enumerate(MODALITIES):
                    if not present_np[:, mi].any():
                        continue
                    if condition.family == "context" and m != condition.affected:
                        continue
                    removed = present.clone()
                    removed[:, mi] = 0
                    without = predict(method, models, xs, removed, raw, cal, alphas).cpu().numpy()
                    loss_without = -np.log(np.maximum(without[np.arange(len(y)), y], 1e-12))
                    score_type = "native" if method == "tmc" else ("calibrated" if method.endswith("calibrated") else "raw")
                    score = {"raw": raw_r, "calibrated": cal_r, "native": native_r}[score_type][:, mi]
                    frame = pd.DataFrame({
                        "sample_id": test["id"], "video_id": test["video"], "y": y.astype(np.int8),
                        "seed": seed, "method": method, "modality": m,
                        **asdict(condition), "condition": condition.key,
                        "r": score, "r_raw": raw_r[:, mi], "r_calibrated": cal_r[:, mi],
                        "entropy_score": raw_h[:, mi], "r_native": native_r[:, mi],
                        "threshold": float(thresholds[score_type][mi]),
                        "threshold_calibrated": float(thresholds["calibrated"][mi]),
                        "high": score >= thresholds[score_type][mi],
                        "high_calibrated": cal_r[:, mi] >= thresholds["calibrated"][mi],
                        "full_p0": full[:, 0], "full_p1": full[:, 1],
                        "without_p0": without[:, 0], "without_p1": without[:, 1],
                        "mono_raw_p1": mono_raw[:, mi, 1], "mono_cal_p1": mono_cal[:, mi, 1],
                        "tmc_alpha0": native_alpha[:, mi, 0], "tmc_alpha1": native_alpha[:, mi, 1],
                        "loss_full": loss_full, "loss_without": loss_without,
                        "u": loss_without-loss_full,
                        "negative_flip": (without.argmax(1) == y) & (full.argmax(1) != y),
                        "positive_flip": (without.argmax(1) != y) & (full.argmax(1) == y),
                    })
                    frames.append(frame)
                table = pa.Table.from_pandas(pd.concat(frames, ignore_index=True), preserve_index=False)
                if method not in writers:
                    writers[method] = pq.ParquetWriter(output / f"{method}.parquet", table.schema, compression="zstd")
                writers[method].write_table(table)
            if ci % 20 == 0:
                print(f"seed={seed}: evaluated condition {ci+1}, {condition.key}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    write_json(output / "evaluation_complete.json", {"seed": seed, "conditions": ci+1, "methods": list(METHODS)})
