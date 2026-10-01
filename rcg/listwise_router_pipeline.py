"""Train and evaluate the cross-fitted analytic-residual coalition router."""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .io import load_json, write_json
from .listwise_router import (AnalyticResidualListwiseRouter,
                              anchor_preserving_candidates, candidate_recall,
                              build_soft_coalition_targets,
                              listwise_router_objective, ranking_metrics)
from .posterior_analytic_pipeline import _load_backbone
from .projected_fusion_pipeline import load_posteriors
from .rcg_fusion import nonempty_coalitions
from .rcg_fusion_pipeline import (attach_observed_losses, load_dataset,
                                  predict_outputs, seed_all)


DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def tensors(split, device):
    return ([torch.as_tensor(x, dtype=torch.float32, device=device) for x in split["x"]],
            torch.as_tensor(split["y"], dtype=torch.long, device=device))


def target_tensors(target, device):
    keys = ("soft_oracle", "pairwise_preference", "stable_improvement",
            "agreement_weight")
    return {key: torch.as_tensor(target[key], dtype=torch.float32, device=device)
            for key in keys}


def train_router(model, train_split, train_probabilities, train_target,
                 selection_split, selection_probabilities, selection_target,
                 masks, device, *, seed, epochs=100, batch_size=64, patience=10):
    seed_all(seed)
    tx, ty = tensors(train_split, device); vx, vy = tensors(selection_split, device)
    tcp = torch.as_tensor(train_probabilities, dtype=torch.float32, device=device)
    if tcp.ndim == 3:
        tcp = tcp[None]
    if tcp.ndim != 4:
        raise ValueError("train probabilities must be [teacher,sample,coalition,class]")
    vcp = torch.as_tensor(selection_probabilities, dtype=torch.float32, device=device)
    mt = torch.as_tensor(masks, dtype=torch.float32, device=device)
    tt, vt = target_tensors(train_target, device), target_tensors(selection_target, device)
    train_posterior = train_target.get("posterior_override")
    selection_posterior = selection_target.get("posterior_override")
    train_posterior = (None if train_posterior is None else
                       torch.as_tensor(train_posterior, dtype=torch.float32, device=device))
    selection_posterior = (None if selection_posterior is None else
                           torch.as_tensor(selection_posterior, dtype=torch.float32, device=device))
    posterior_weight = 0. if train_posterior is not None else 1.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train(); train_values = []
        for index in torch.randperm(len(ty), device=device).split(batch_size):
            teachers = len(tcp)
            repeated = index[None].expand(teachers, -1).reshape(-1)
            coalition = tcp[:, index].reshape(-1, *tcp.shape[2:])
            if train_posterior is None:
                override = None
            elif train_posterior.ndim == 3:
                override = train_posterior[:, index].reshape(-1, train_posterior.shape[-1])
            else:
                override = train_posterior[index]
            output = model([x[repeated] for x in tx], coalition, mt,
                           posterior_override=override)
            batch_target = {key: value[index][None].expand(
                teachers, *value[index].shape).reshape(-1, *value.shape[1:])
                for key, value in tt.items()}
            loss, _ = listwise_router_objective(
                output, ty[repeated], batch_target, posterior_weight=posterior_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            train_values.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            output = model(vx, vcp, mt, posterior_override=selection_posterior)
            loss, parts = listwise_router_objective(
                output, vy, vt, posterior_weight=posterior_weight)
        value = float(loss)
        history.append({"epoch": epoch+1, "train": float(np.mean(train_values)),
                        "selection": value,
                        **{f"selection_{key}": float(part) for key, part in parts.items()}})
        if value < best-1e-6:
            best, state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if state is None:
        raise RuntimeError("router training produced no checkpoint")
    model.load_state_dict(state); model.eval()
    return history


@torch.inference_mode()
def predict_router(model, split, probabilities, masks, device, posterior_override=None):
    xs, _ = tensors(split, device)
    override = (None if posterior_override is None else
                torch.as_tensor(posterior_override, dtype=torch.float32, device=device))
    output = model(xs, torch.as_tensor(probabilities, dtype=torch.float32, device=device),
                   torch.as_tensor(masks, dtype=torch.float32, device=device),
                   posterior_override=override)
    return {key: value.detach().cpu().numpy() for key, value in output.items()
            if key != "full_index"}


def select_residual_blend(output, losses, grid=(0., .1, .25, .5, .75, 1.)):
    """Choose residual strength on selection regret; zero is a safe analytic fallback."""
    losses = np.asarray(losses); rows = np.arange(len(losses)); oracle = losses.min(1)
    records = []
    for value in grid:
        score = output["analytic_score"]+float(value)*output["residual"]
        chosen = score.argmax(1)
        metrics = ranking_metrics(score, losses)
        records.append({"residual_blend": float(value),
                        "selection_regret": float((losses[rows, chosen]-oracle).mean()),
                        **metrics})
    # Regret is the deployment objective.  Ties prefer the smaller residual.
    best = min(records, key=lambda item: (item["selection_regret"],
                                          item["residual_blend"]))
    return best["residual_blend"], records


def apply_residual_blend(output, value):
    result = dict(output)
    result["score"] = output["analytic_score"]+float(value)*output["residual"]
    return result


def evaluate(output, bundle, labels):
    learned = ranking_metrics(output["score"], bundle["losses"])
    analytic = ranking_metrics(output["analytic_score"], bundle["losses"])
    rows = np.arange(len(labels)); chosen = output["score"].argmax(1)
    analytic_chosen = output["analytic_score"].argmax(1)
    oracle = bundle["losses"].argmin(1); full = bundle["losses"].shape[1]-1
    full_loss = bundle["losses"][:, full]
    selected_loss = bundle["losses"][rows, chosen]
    analytic_loss = bundle["losses"][rows, analytic_chosen]
    probability = bundle["probabilities"][rows, chosen]
    anchored = anchor_preserving_candidates(
        output["analytic_score"], output["score"],
        top_k=min(3, bundle["losses"].shape[1]))
    return {
        **{f"router_{key}": value for key, value in learned.items()},
        **{f"analytic_{key}": value for key, value in analytic.items()},
        "top1_gain": learned["top1"]-analytic["top1"],
        "top2_gain": learned["top2"]-analytic["top2"],
        "selected_nll": float(selected_loss.mean()),
        "analytic_selected_nll": float(analytic_loss.mean()),
        "full_nll": float(full_loss.mean()),
        "oracle_nll": float(bundle["losses"].min(1).mean()),
        "selection_regret": float((selected_loss-bundle["losses"].min(1)).mean()),
        "analytic_selection_regret": float(
            (analytic_loss-bundle["losses"].min(1)).mean()),
        "full_regret": float((full_loss-bundle["losses"].min(1)).mean()),
        "selected_accuracy": float((probability.argmax(1) == labels).mean()),
        "full_accuracy": float(
            (bundle["probabilities"][:, full].argmax(1) == labels).mean()),
        "oracle_full_rate": float((oracle == full).mean()),
        "posterior_accuracy": float((output["posterior"].argmax(1) == labels).mean()),
        "anchored_top1": candidate_recall(anchored[:, :1], bundle["losses"]),
        "anchored_top2": candidate_recall(anchored[:, :2], bundle["losses"]),
        "anchored_top3": candidate_recall(anchored, bundle["losses"]),
    }, chosen, oracle


def run(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    folds, names = load_dataset(args.dataset, args.data)
    base, root = Path(args.base_run), Path(args.output); root.mkdir(parents=True, exist_ok=True)
    posterior_root = None if args.posterior_run is None else Path(args.posterior_run)
    manifest = {"dataset": args.dataset, "base_run": str(base.resolve()),
                "seeds": args.seeds, "oracle_temperature": args.oracle_temperature,
                "stable_fraction": args.stable_fraction, "epsilon": args.epsilon,
                "posterior_run": None if posterior_root is None else str(posterior_root.resolve()),
                "objective": "OOF soft oracle + pairwise preference + stable improvement + posterior",
                "test_labels_role": "evaluation only", "outer_folds": len(folds)}
    if (root/"manifest.json").exists() and load_json(root/"manifest.json") != manifest:
        raise ValueError("manifest differs; use a new output directory")
    write_json(root/"manifest.json", manifest)
    metrics, start = [], time.perf_counter()
    for fold_index, splits in enumerate(folds):
        fold_base = base/f"fold_{fold_index}"; fold_out = root/f"fold_{fold_index}"
        fold_out.mkdir(exist_ok=True)
        with np.load(fold_base/"oof_targets.npz") as saved:
            oof = {key: saved[key] for key in saved.files}
        masks = nonempty_coalitions(len(names))
        train_target = build_soft_coalition_targets(
            oof["losses"], oracle_temperature=args.oracle_temperature,
            epsilon=args.epsilon, stable_fraction=args.stable_fraction)
        target_path = fold_out/"soft_coalition_targets.npz"
        if not target_path.exists():
            np.savez_compressed(
                target_path, sample_id=splits["train"]["id"],
                group_id=splits["train"]["group"], teacher_fold=oof["fold"],
                teacher_seeds=oof["teacher_seeds"], **train_target)
        for task_seed in args.seeds:
            out = fold_out/f"seed_{task_seed}"; out.mkdir(exist_ok=True)
            if (out/"complete.json").exists():
                item = load_json(out/"metrics.json")
                item.setdefault("n_test", len(splits["test"]["y"])); metrics.append(item); continue
            backbone, temperatures, dims, classes = _load_backbone(
                fold_base/f"seed_{task_seed}"/"backbone.pt", device)
            bundles = {}
            for split_name in ("selection", "test"):
                raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
                bundles[split_name] = attach_observed_losses(raw, splits[split_name]["y"])
            # Every OOF teacher context is a training view.  Targets remain the
            # teacher aggregate and each original sample has total weight one.
            train_probability = oof["probabilities"]
            selection_target = build_soft_coalition_targets(
                bundles["selection"]["losses"][None],
                oracle_temperature=args.oracle_temperature, epsilon=args.epsilon,
                stable_fraction=args.stable_fraction)
            posterior = {}
            if posterior_root is not None:
                train_members = []
                for teacher_probability in train_probability:
                    _, teacher_posterior = load_posteriors(
                        posterior_root, task_seed, dims, classes, masks, splits["train"],
                        teacher_probability, device, fold_index=fold_index)
                    train_members.append(teacher_posterior)
                posterior["train"] = np.stack(train_members)
                for split_name in ("selection", "test"):
                    _, posterior[split_name] = load_posteriors(
                        posterior_root, task_seed, dims, classes, masks, splits[split_name],
                        bundles[split_name]["probabilities"], device, fold_index=fold_index)
                train_target["posterior_override"] = posterior["train"]
                selection_target["posterior_override"] = posterior["selection"]
            seed_all(task_seed+fold_index*10000)
            model = AnalyticResidualListwiseRouter(
                dims, classes, len(masks), residual_scale=args.residual_scale).to(device)
            history = train_router(
                model, splits["train"], train_probability, train_target,
                splits["selection"], bundles["selection"]["probabilities"], selection_target,
                masks, device, seed=task_seed+fold_index*10000, epochs=args.epochs,
                batch_size=args.batch_size)
            selection_output = predict_router(
                model, splits["selection"], bundles["selection"]["probabilities"], masks, device,
                posterior.get("selection"))
            residual_blend, residual_grid = select_residual_blend(
                selection_output, bundles["selection"]["losses"])
            output = predict_router(
                model, splits["test"], bundles["test"]["probabilities"], masks, device,
                posterior.get("test"))
            output = apply_residual_blend(output, residual_blend)
            item, chosen, oracle = evaluate(output, bundles["test"], splits["test"]["y"])
            item.update({"dataset": args.dataset, "fold": fold_index,
                         "train_seed": task_seed, "n_test": len(chosen),
                         "residual_blend": residual_blend})
            metrics.append(item); write_json(out/"metrics.json", item)
            torch.save({"state_dict": model.state_dict(), "dims": dims, "classes": classes,
                        "n_coalitions": len(masks), "residual_scale": args.residual_scale},
                       out/"listwise_router.pt")
            write_json(out/"training.json", {"epochs": history,
                                               "residual_blend_grid": residual_grid,
                                               "selected_residual_blend": residual_blend})
            frame = pd.DataFrame({
                "sample_id": splits["test"]["id"],
                "group_or_video_id": splits["test"]["group"],
                "label": splits["test"]["y"], "chosen_coalition": chosen,
                "oracle_coalition": oracle, "train_seed": task_seed, "fold": fold_index})
            anchored = anchor_preserving_candidates(
                output["analytic_score"], output["score"], top_k=min(3, len(masks)))
            for rank in range(anchored.shape[1]):
                frame[f"anchored_candidate_{rank+1}"] = anchored[:, rank]
            for index in range(len(masks)):
                frame[f"score_{index}"] = output["score"][:, index]
                frame[f"analytic_score_{index}"] = output["analytic_score"][:, index]
                frame[f"observed_loss_{index}"] = bundles["test"]["losses"][:, index]
            frame.to_parquet(out/"predictions.parquet", index=False)
            write_json(out/"complete.json", {"n_test": len(frame)})
            print(f"{args.dataset} fold={fold_index} seed={task_seed}: "
                  f"top1={item['router_top1']:.3f} top2={item['router_top2']:.3f} "
                  f"gain={item['top1_gain']:.3f}", flush=True)
    fold_table = pd.DataFrame(metrics); fold_table.to_csv(root/"metrics_by_fold_seed.csv", index=False)
    numeric = fold_table.select_dtypes(include=[np.number]).columns.difference(
        ["fold", "train_seed", "n_test"])
    seed_rows = []
    for seed, values in fold_table.groupby("train_seed"):
        weights = values.n_test.to_numpy(float)
        row = {"dataset": args.dataset, "train_seed": int(seed), "n_test": int(weights.sum())}
        row.update({column: float(np.average(values[column], weights=weights)) for column in numeric})
        seed_rows.append(row)
    table = pd.DataFrame(seed_rows); table.to_csv(root/"metrics_by_seed.csv", index=False)
    summary_numeric = table.select_dtypes(include=[np.number]).columns.difference(
        ["train_seed", "n_test"])
    pd.DataFrame({"metric": summary_numeric,
                  "mean": [table[c].mean() for c in summary_numeric],
                  "std": [table[c].std(ddof=1) for c in summary_numeric]}).to_csv(
                      root/"metrics_summary.csv", index=False)
    mean = table.select_dtypes(include=[np.number]).mean()
    checks = {"top1_gain_10pp": mean.top1_gain >= .10,
              "top2_at_least_70pct": mean.router_top2 >= .70,
              "pairwise_better_than_analytic":
                  mean.router_pairwise_accuracy > mean.analytic_pairwise_accuracy}
    write_json(root/"complete.json", {"checks": checks, "passed": all(checks.values()),
                                      "wall_seconds": time.perf_counter()-start})


def main():
    parser = argparse.ArgumentParser(description="OOF analytic-residual listwise coalition router")
    parser.add_argument("--dataset", choices=("mosi", "mosei", "cremad", "avmnist"), required=True)
    parser.add_argument("--data", default="data"); parser.add_argument("--base-run", required=True)
    parser.add_argument("--posterior-run",
                        help="optional frozen posterior ensemble used as the analytic prior")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--oracle-temperature", type=float, default=.1)
    parser.add_argument("--stable-fraction", type=float, default=.8)
    parser.add_argument("--epsilon", type=float, default=.01)
    parser.add_argument("--residual-scale", type=float, default=.5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
