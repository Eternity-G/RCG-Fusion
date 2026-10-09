"""Coalition-valid training and sample-level diagnostics for strong baselines."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F

from .strong_models import build_model


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def all_masks(n, include_empty=False):
    result = [(0,)*n] if include_empty else []
    for size in range(1, n+1):
        result.extend(tuple(int(i in subset) for i in range(n)) for subset in combinations(range(n), size))
    return tuple(result)


def mask_name(mask, names):
    return "+".join(name for name, keep in zip(names, mask) if keep) or "empty"


def draw_masks(n_rows, n_modalities, device):
    masks = torch.tensor(all_masks(n_modalities), dtype=torch.float32, device=device)
    choice = masks[torch.randint(0, len(masks)-1, (n_rows,), device=device)]
    full = torch.ones((n_rows, n_modalities), device=device)
    return torch.where((torch.rand(n_rows, 1, device=device)<.5), full, choice)


def fit_temperature(logits, labels):
    logits = np.asarray(logits, dtype=np.float64); labels = np.asarray(labels)
    def objective(log_t):
        scaled = logits/np.exp(log_t)
        return float((logsumexp(scaled, axis=1)-scaled[np.arange(len(labels)), labels]).mean())
    result = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
    return float(np.exp(result.x))


def split_tensor(split, device):
    return ([torch.as_tensor(x, dtype=torch.float32, device=device) for x in split["x"]],
            torch.as_tensor(split["y"], dtype=torch.long, device=device))


def train(model, train_split, select_split, seed, *, epochs=100, batch_size=64,
          lr=1e-3, weight_decay=.01, patience=10, aux_weight=1.0):
    seed_all(seed)
    device = next(model.parameters()).device
    xs, y = split_tensor(train_split, device); vx, vy = split_tensor(select_split, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train(); epoch_loss = []
        for idx in torch.randperm(len(y), device=device).split(batch_size):
            bx, by = [x[idx] for x in xs], y[idx]
            mask = draw_masks(len(idx), len(xs), device)
            optimizer.zero_grad(set_to_none=True)
            output = model(bx, mask)
            loss = F.cross_entropy(output["logits"], by)
            if hasattr(model, "auxiliary"):
                loss = loss + aux_weight*model.auxiliary(output, by, mask)
            if hasattr(model, "interaction_regularizer"):
                full_rows=mask.bool().all(1)
                if full_rows.any():
                    loss = loss + .01*model.interaction_regularizer([x[full_rows] for x in bx],mask[full_rows])
            if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite {model.method} loss")
            loss.backward(); optimizer.step(); epoch_loss.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            full = torch.ones((len(vy), len(vx)), device=device)
            value = float(F.cross_entropy(model(vx, full)["logits"], vy))
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(epoch_loss)), "selection_nll": value})
        if value < best-1e-6:
            best, state, stale = value, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if stale >= patience: break
    model.load_state_dict(state); model.eval()
    return history


def exact_shapley(losses, n_modalities):
    n = len(next(iter(losses.values())))
    result = np.zeros((n, n_modalities), dtype=np.float64)
    for m in range(n_modalities):
        others = [i for i in range(n_modalities) if i != m]
        for size in range(len(others)+1):
            weight = math.factorial(size)*math.factorial(n_modalities-size-1)/math.factorial(n_modalities)
            for subset in combinations(others, size):
                before = tuple(int(i in subset) for i in range(n_modalities))
                after = list(before); after[m] = 1
                result[:, m] += weight*(losses[before]-losses[tuple(after)])
    return result


@torch.inference_mode()
def calibration(model, split, masks):
    device = next(model.parameters()).device; xs, y = split_tensor(split, device)
    temps, scores, native_temps = {}, [], None
    for mask in masks:
        present = torch.tensor(mask, dtype=torch.float32, device=device)[None].repeat(len(y), 1)
        output = model(xs, present)
        temps[mask] = fit_temperature(output["logits"].cpu().numpy(), y.cpu().numpy())
        if all(mask):
            score=output["native_score"].cpu().numpy()
            if model.method == "coalition_concat":
                mono=output["mono_logits"].cpu().numpy(); native_temps=[]
                for i in range(mono.shape[1]): native_temps.append(fit_temperature(mono[:,i],y.cpu().numpy()))
                score=np.stack([torch.softmax(torch.tensor(mono[:,i]/native_temps[i]),-1).max(1).values.numpy()
                                for i in range(mono.shape[1])],1)
            scores.append(score)
    score = scores[0]
    thresholds = [float(np.quantile(score[:, i], .9)) for i in range(score.shape[1])]
    return temps, thresholds, native_temps


@torch.inference_mode()
def evaluate(model, split, dataset, method, fold, seed, names, temps, thresholds, prior, native_temps=None):
    device = next(model.parameters()).device; xs, y_t = split_tensor(split, device)
    y = y_t.cpu().numpy(); n_modalities = len(names); masks = all_masks(n_modalities)
    empty = (0,)*n_modalities
    losses = {empty: -np.log(np.maximum(prior[y], 1e-12))}; probabilities = {}; native = {}
    extra_scores = {}
    for mask in masks:
        present = torch.tensor(mask, dtype=torch.float32, device=device)[None].repeat(len(y), 1)
        output = model(xs, present)
        prob = torch.softmax(output["logits"]/temps[mask], -1).cpu().numpy()
        probabilities[mask] = prob
        losses[mask] = -np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12))
        if all(mask):
            native = output["native_score"].cpu().numpy()
            if method == "concat" and native_temps is not None:
                mono=output["mono_logits"].cpu().numpy()
                native=np.stack([torch.softmax(torch.tensor(mono[:,i]/native_temps[i]),-1).max(1).values.numpy()
                                 for i in range(mono.shape[1])],1)
            for key in ("mono_confidence", "holo_confidence", "cobelief", "fusion_weight", "route"):
                if key in output: extra_scores[key] = output[key].cpu().numpy()
    shapley = exact_shapley(losses, n_modalities)
    full = (1,)*n_modalities; full_loss = losses[full]
    stack = np.stack([losses[m] for m in masks], 1); best = stack.argmin(1)
    oracle = np.array([mask_name(masks[i], names) for i in best])
    full_pred = probabilities[full].argmax(1)
    oracle_prob = np.stack([probabilities[masks[i]][row] for row, i in enumerate(best)])
    oracle_pred = oracle_prob.argmax(1)
    frame = pd.DataFrame({
        "dataset": dataset, "method": method, "fold": fold, "train_seed": seed,
        "sample_id": split["id"], "group_or_video_id": split["group"], "label": y,
        "full_loss": full_loss, "fusion_regret": full_loss-stack.min(1),
        "full_prediction": full_pred, "oracle_prediction": oracle_pred,
        "oracle_coalition": oracle, "full_correct": full_pred == y, "oracle_correct": oracle_pred == y,
    })
    for cls in range(probabilities[full].shape[1]):
        frame[f"full_p{cls}"] = probabilities[full][:, cls]
        frame[f"oracle_p{cls}"] = oracle_prob[:, cls]
    for i, name in enumerate(names):
        without = list(full); without[i] = 0; without = tuple(without)
        deletion = losses[without]-full_loss
        without_pred = probabilities[without].argmax(1)
        frame[f"native_score_{name}"] = native[:, i]
        frame[f"high_reliability_{name}"] = native[:, i] >= thresholds[i]
        frame[f"deletion_contribution_{name}"] = deletion
        frame[f"shapley_contribution_{name}"] = shapley[:, i]
        frame[f"negative_flip_{name}"] = (without_pred == y) & (full_pred != y)
        frame[f"correction_by_deletion_{name}"] = (full_pred != y) & (without_pred == y)
        for key, values in extra_scores.items():
            if values.ndim == 2 and i < values.shape[1]: frame[f"{key}_{name}"] = values[:, i]
    for mask in masks:
        cname = mask_name(mask, names)
        frame[f"loss_{cname}"] = losses[mask]
    reliability_json=[json.dumps({name:float(v) for name,v in zip(names,row)},separators=(",",":")) for row in native]
    singleton={name:probabilities[tuple(int(j==i) for j in range(n_modalities))] for i,name in enumerate(names)}
    unimodal_json=[json.dumps({name:singleton[name][row].tolist() for name in names},separators=(",",":")) for row in range(len(y))]
    contribution_json=[json.dumps({name:float(frame.iloc[row][f"deletion_contribution_{name}"]) for name in names},separators=(",",":")) for row in range(len(frame))]
    shapley_json=[json.dumps({name:float(frame.iloc[row][f"shapley_contribution_{name}"]) for name in names},separators=(",",":")) for row in range(len(frame))]
    long=[]
    for mask in masks:
        cname=mask_name(mask,names); prob=probabilities[mask]
        long.append(pd.DataFrame({
            "dataset":dataset,"sample_id":split["id"],"group_or_video_id":split["group"],"train_seed":seed,
            "fold":fold,"method":method,"corruption_type":"clean","corruption_level":0.,"corruption_seed":-1,
            "available_modalities":mask_name(full,names),"coalition":cname,
            "unimodal_probabilities":unimodal_json,
            "reliability_scores":reliability_json,"predicted_coalition_loss":np.nan,
            "predicted_loss_uncertainty":np.nan,"observed_coalition_loss":losses[mask],
            "conditional_contribution":contribution_json,"shapley_contribution":shapley_json,
            "selected_coalition":oracle,"final_probability":[json.dumps(row.tolist(),separators=(",",":")) for row in prob],
            "label":y,
        }))
    return frame,pd.concat(long,ignore_index=True)


def normalize_splits(splits):
    n_mod = len(splits["train"]["x"])
    for m in range(n_mod):
        train = splits["train"]["x"][m]
        mean = train.mean(0, dtype=np.float64).astype(np.float32)
        std = train.std(0, dtype=np.float64).astype(np.float32); std[std < 1e-6] = 1
        for split in splits.values(): split["x"][m] = ((split["x"][m]-mean)/std).astype(np.float32)
    return splits


def cremadsplits(path, split_seed=2026):
    with np.load(path) as data:
        x = [data["visual"].copy(), data["audio"].copy()]; y=data["label"].copy()
        ids=data["sample_id"].copy(); groups=data["actor"].copy()
    output = []
    for fold, (remaining, test) in enumerate(GroupKFold(5).split(y, y, groups)):
        actors = np.unique(groups[remaining]); rng=np.random.default_rng(split_seed+fold); rng.shuffle(actors)
        n_hold=max(1, round(len(actors)*.125)); select_actors=actors[:n_hold]; calibration_actors=actors[n_hold:2*n_hold]
        selection=remaining[np.isin(groups[remaining], select_actors)]
        calibration_idx=remaining[np.isin(groups[remaining], calibration_actors)]
        train_idx=remaining[~np.isin(groups[remaining], np.r_[select_actors, calibration_actors])]
        def take(index): return {"x":[z[index].copy() for z in x], "y":y[index], "id":ids[index], "group":groups[index]}
        output.append(normalize_splits({"train":take(train_idx), "selection":take(selection),
                                        "calibration":take(calibration_idx), "test":take(test)}))
    return output, ("visual", "audio")


def mmsasplits(root, dataset):
    source=Path(root)/dataset/"prepared"; raw={}
    names=("text", "audio", "vision")
    for key in ("train", "selection", "calibration", "test"):
        with np.load(source/f"{key}.npz") as d:
            raw[key]={"x":[d[n].copy() for n in names], "y":d["y"].copy(), "id":d["id"].copy(), "group":d["video"].copy()}
    return [raw], names


def avmnistsplits(root):
    root = Path(root)
    if (root/"avmnist_hf"/"prepared").exists():
        root = root/"avmnist_hf"/"prepared"
    elif (root/"avmnist"/"prepared").exists():
        root = root/"avmnist"/"prepared"
    elif (root/"prepared").exists():
        root = root/"prepared"
    raw = {}
    names = ("image", "audio")
    for key in ("train", "selection", "calibration", "test"):
        with np.load(root/f"{key}.npz") as data:
            ids = data["id"].copy()
            raw[key] = {"x": [data[name].copy() for name in names],
                        "y": data["y"].copy(), "id": ids,
                        "group": ids.copy()}
    return [normalize_splits(raw)], names


def run(dataset, data, output, methods, seeds, device="cuda", epochs=100):
    device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    if dataset == "cremad":
        folds, names = cremadsplits(data)
    elif dataset == "avmnist":
        folds, names = avmnistsplits(data)
    else:
        folds, names = mmsasplits(data, dataset)
    output=Path(output); output.mkdir(parents=True, exist_ok=True)
    for fold, splits in enumerate(folds):
        dims=[x.shape[1] for x in splits["train"]["x"]]; classes=int(max(s["y"].max() for s in splits.values())+1)
        prior=np.bincount(splits["train"]["y"], minlength=classes); prior=prior/prior.sum()
        for method in methods:
            for seed in seeds:
                target=output/dataset/f"fold_{fold}"/method/f"seed_{seed}"; target.mkdir(parents=True, exist_ok=True)
                record=target/"samples.parquet"
                if record.exists() and (target/"coalitions.parquet").exists():
                    print(f"skip complete {dataset} fold={fold} method={method} seed={seed}", flush=True); continue
                seed_all(seed+fold*1000); model=build_model(method, dims, classes).to(device)
                start=time.perf_counter(); history=train(model, splits["train"], splits["selection"], seed+fold*1000, epochs=epochs)
                masks=all_masks(len(names)); temps, thresholds, native_temps=calibration(model, splits["calibration"], masks)
                frame,long_frame=evaluate(model, splits["test"], dataset, method, fold, seed, names, temps, thresholds, prior, native_temps)
                frame.to_parquet(record, index=False)
                long_frame.to_parquet(target/"coalitions.parquet",index=False)
                torch.save(model.state_dict(), target/"model.pt")
                (target/"run.json").write_text(json.dumps({"dataset":dataset,"method":method,"fold":fold,"seed":seed,
                    "names":names,"dims":dims,"classes":classes,"temperatures":{mask_name(k,names):v for k,v in temps.items()},
                    "thresholds":dict(zip(names,thresholds)),"native_temperatures":native_temps,
                    "history":history,"seconds":time.perf_counter()-start}, indent=2), encoding="utf-8")
                print(f"done {dataset} fold={fold} method={method} seed={seed} n={len(frame)}", flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset", choices=("mosi","mosei","cremad","avmnist"), required=True)
    p.add_argument("--data", type=Path, default=Path("data")); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=["concat","tmc","qmf","pdf","i2moe"])
    p.add_argument("--seeds", nargs="+", type=int, default=[11,22,33,44,55]); p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=100); args=p.parse_args()
    data=args.data/"cremad_wav2vec2_r3d18.npz" if args.dataset=="cremad" and args.data.is_dir() else args.data
    run(args.dataset, data, args.output, args.methods, args.seeds, args.device, args.epochs)


if __name__ == "__main__": main()
