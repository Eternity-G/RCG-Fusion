from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.nn import functional as F

from . import MODALITIES
from .io import write_json
from .models import Head, Concat, TMC, draw_presence, ds_fusion, evidential_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def tensors(split, device):
    return [torch.as_tensor(split[m], dtype=torch.float32, device=device) for m in MODALITIES], torch.as_tensor(split["y"], dtype=torch.long, device=device)


def fit_temperature(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    from scipy.special import logsumexp
    def objective(log_t):
        scaled = logits / np.exp(log_t)
        return float((logsumexp(scaled, axis=1)-scaled[np.arange(len(labels)), labels]).mean())
    result = minimize_scalar(objective, bounds=(-4, 4), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(result.x)), {"before_nll": objective(0), "after_nll": result.fun,
                                     "boundary": bool(abs(result.x) > 3.99)}


def train_one(model, kind, train, valid, cfg, modality=None):
    xs, y = train
    vx, vy = valid
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        losses = []
        for index in torch.randperm(len(y), device=y.device).split(cfg["batch_size"]):
            bx, by = [x[index] for x in xs], y[index]
            optimizer.zero_grad(set_to_none=True)
            present = torch.ones((len(index), 3), device=y.device)
            if kind == "unimodal":
                loss = F.cross_entropy(model(bx[modality]), by)
            elif kind.startswith("concat"):
                if kind == "concat_masked":
                    present = draw_presence(len(index), y.device)
                loss = F.cross_entropy(model(bx, present), by)
            else:
                alphas = model(bx)
                fused = ds_fusion(alphas, present)
                anneal = min(1.0, (epoch+1)/cfg["tmc_anneal_epochs"])
                loss = evidential_loss(fused, by, anneal)
                loss = loss + sum(evidential_loss(alphas[:, m], by, anneal) for m in range(3))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {kind} training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            present = torch.ones((len(vy), 3), device=y.device)
            if kind == "unimodal":
                vloss = F.cross_entropy(model(vx[modality]), vy)
            elif kind.startswith("concat"):
                vloss = F.cross_entropy(model(vx, present), vy)
            else:
                a = ds_fusion(model(vx), present)
                vloss = F.nll_loss((a/a.sum(1, keepdim=True)).clamp_min(1e-12).log(), vy)
        value = float(vloss)
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)), "selection_nll": value})
        if value < best-1e-6:
            best, state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(state)
    model.eval()
    return history


def build_models(dims, cfg, device):
    args = {"hidden": cfg["hidden"], "dropout": cfg["dropout"]}
    result = {f"uni_{m}": Head(d, **args).to(device) for m, d in zip(MODALITIES, dims)}
    result.update({k: Concat(dims, **args).to(device) for k in ("concat_masked", "concat_clean")})
    result["tmc"] = TMC(dims, **args).to(device)
    return result


def train_seed(splits, cfg, seed, output, device):
    seed_all(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dims = [splits["train"][m].shape[1] for m in MODALITIES]
    models = build_models(dims, cfg, device)
    train, valid = tensors(splits["train"], device), tensors(splits["selection"], device)
    logs, start = {}, time.perf_counter()
    for i, (name, model) in enumerate(models.items()):
        seed_all(seed * 100 + i)
        modality = list(MODALITIES).index(name[4:]) if name.startswith("uni_") else None
        kind = "unimodal" if modality is not None else name
        logs[name] = train_one(model, kind, train, valid, cfg, modality)
        print(f"seed={seed} {name}: {len(logs[name])} epochs, best NLL={min(x['selection_nll'] for x in logs[name]):.4f}", flush=True)
    cx, cy = tensors(splits["calibration"], device)
    temps, calibration = {}, {}
    with torch.no_grad():
        for m in MODALITIES:
            logits = models[f"uni_{m}"](cx[list(MODALITIES).index(m)]).cpu().numpy()
            temps[m], calibration[m] = fit_temperature(logits, cy.cpu().numpy())
    torch.save({"models": {k: v.state_dict() for k, v in models.items()}, "temperatures": temps,
                "dims": dims, "seed": seed}, output / "checkpoint.pt")
    write_json(output / "training.json", {"history": logs, "calibration": calibration,
                                           "temperatures": temps, "seconds": time.perf_counter()-start})
    return models, temps


def load_models(path, cfg, device):
    state = torch.load(path, map_location=device, weights_only=True)
    models = build_models(state["dims"], cfg, device)
    for key, model in models.items():
        model.load_state_dict(state["models"][key])
        model.eval()
    return models, state["temperatures"]
