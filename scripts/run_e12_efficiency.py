"""E12: reproducible parameter, latency, memory, and coalition-cost benchmark.

The benchmark covers the frozen-feature fusion stack.  Feature encoders are
excluded because they are precomputed for every method in this study.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from rcg.anchored_mixer import AnchoredCandidateMixer
from rcg.listwise_router import AnalyticResidualListwiseRouter
from rcg.posterior_analytic import RelationalPosteriorEstimator
from rcg.rcg_fusion import CoalitionAwareBackbone, evaluate_all_coalitions, nonempty_coalitions
from rcg.rcg_fusion_pipeline import load_dataset


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_member(dataset, seed, fold, device):
    roots = {
        "mosi": ("rcg-fusion-mosi-v4", "rcg-posterior-analytic-mosi-v2", "formal-e5-mosi"),
        "mosei": ("rcg-fusion-mosei-shrinkage-v1",
                  "rcg-posterior-analytic-mosei-shrinkage-v1", "formal-e5-mosei"),
        "cremad": ("rcg-fusion-cremad-v1", "rcg-posterior-analytic-cremad-v1",
                   "formal-e5-cremad"),
        "avmnist": ("rcg-fusion-avmnist-v1", "rcg-posterior-analytic-avmnist-v1",
                    "formal-e5-avmnist"),
    }
    base_name, posterior_name, e5_name = roots[dataset]
    base_root, posterior_root, e5_root = (ROOT/"runs"/base_name,
                                          ROOT/"runs"/posterior_name,
                                          ROOT/"runs"/e5_name)
    base = torch.load(base_root/f"fold_{fold}/seed_{seed}/backbone.pt",
                      map_location=device, weights_only=False)
    backbone = CoalitionAwareBackbone(base["dims"], base["classes"]).to(device)
    backbone.load_state_dict(base["state_dict"]); backbone.eval()
    masks = nonempty_coalitions(len(base["dims"]))
    mask_tensor = torch.as_tensor(masks, dtype=torch.float32, device=device)
    posterior_models = []
    for path in sorted((posterior_root/f"fold_{fold}/seed_{seed}").glob("posterior_*.pt")):
        saved = torch.load(path, map_location=device, weights_only=False)
        model = RelationalPosteriorEstimator(
            base["dims"], base["classes"], len(masks),
            use_relations=saved["use_relations"]).to(device)
        model.load_state_dict(saved["state_dict"]); model.eval(); posterior_models.append(model)
    posterior_metrics = json.loads(
        (posterior_root/f"fold_{fold}/seed_{seed}/metrics.json").read_text(encoding="utf-8"))
    posterior_temperature = float(posterior_metrics["posterior_temperature"])
    router_saved = torch.load(e5_root/f"fold_{fold}/seed_{seed}/router.pt",
                              map_location=device, weights_only=True)
    router = AnalyticResidualListwiseRouter(
        base["dims"], base["classes"], len(masks), residual_scale=.5).to(device)
    router.load_state_dict(router_saved["state_dict"]); router.eval()
    mixer_saved = torch.load(e5_root/f"fold_{fold}/seed_{seed}/anchored_harm_oracle.pt",
                             map_location=device, weights_only=True)
    mixer = AnchoredCandidateMixer(base["classes"], anchor_full=True).to(device)
    mixer.load_state_dict(mixer_saved["state_dict"]); mixer.eval()
    return {"backbone": backbone, "posterior": posterior_models, "router": router,
            "mixer": mixer, "masks": mask_tensor,
            "temperatures": torch.as_tensor(base["temperatures"], dtype=torch.float32,
                                             device=device),
            "posterior_temperature": posterior_temperature,
            "blend": float(router_saved["blend"]), "dims": base["dims"],
            "classes": base["classes"]}


def load_backbone_only(dataset, seed, fold, device):
    root = (ROOT/"runs/rcg-fusion-mosi-v4" if dataset == "mosi"
            else ROOT/"runs/rcg-fusion-cremad-v1")
    saved = torch.load(root/f"fold_{fold}/seed_{seed}/backbone.pt",
                       map_location=device, weights_only=False)
    model = CoalitionAwareBackbone(saved["dims"], saved["classes"]).to(device)
    model.load_state_dict(saved["state_dict"]); model.eval()
    return {"backbone": model,
            "temperatures": torch.as_tensor(saved["temperatures"], dtype=torch.float32,
                                             device=device)}


def calibrated_posterior(member, xs, probabilities):
    values = []
    for model in member["posterior"]:
        p = model(xs, probabilities, member["masks"])["posterior"].clamp_min(1e-8)
        values.append(torch.softmax(p.log()/member["posterior_temperature"], -1))
    return torch.stack(values).mean(0)


def candidate_indices(analytic, score, top_k):
    # Formal E5 excludes full fusion from candidate coalitions; full is retained
    # explicitly as action zero.
    analytic, score = analytic[:, :-1], score[:, :-1]
    anchor = analytic.argmax(1)
    order = score.argsort(1, descending=True)
    result = torch.empty((len(anchor), top_k), dtype=torch.long, device=score.device)
    result[:, 0] = anchor
    for row in range(len(anchor)):
        support = order[row][order[row] != anchor[row]][:top_k-1]
        result[row, 1:] = support
    return result


@torch.inference_mode()
def full_forward(member, xs):
    full = torch.ones((len(xs[0]), len(xs)), dtype=torch.float32, device=xs[0].device)
    return torch.softmax(member["backbone"](xs, full)["logits"]/
                         member["temperatures"][-1], -1)


@torch.inference_mode()
def coalition_forward(member, xs):
    _, probability = evaluate_all_coalitions(
        member["backbone"], xs, member["temperatures"])
    return probability


@torch.inference_mode()
def rcg_forward(member, xs):
    probability = coalition_forward(member, xs)
    posterior = calibrated_posterior(member, xs, probability)
    output = member["router"](xs, probability, member["masks"],
                              posterior_override=posterior)
    score = output["analytic_score"]+member["blend"]*output["residual"]
    candidates = candidate_indices(output["analytic_score"], score,
                                   min(3, probability.shape[1]-1))
    rows = torch.arange(len(probability), device=probability.device)[:, None]
    selected = probability[rows, candidates]
    actions = torch.cat((probability[:, -1:, :], posterior[:, None, :], selected), 1)
    mixture = member["mixer"](actions, posterior)["probability"]
    full = probability[:, -1]
    delta = torch.log(mixture.clamp_min(1e-6)/full.clamp_min(1e-6))
    pi = (posterior*(delta > .01)).sum(1)
    alpha = .75*pi.square()
    return (1-alpha[:, None])*full+alpha[:, None]*mixture


def benchmark(name, function, device, rounds, repeats):
    for _ in range(20):
        function()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize(device); start = time.perf_counter()
        for _ in range(repeats):
            function()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter()-start)*1000/repeats)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    function(); torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)/1024**2
    return {"configuration": name, "latency_ms_mean": float(np.mean(samples)),
            "latency_ms_std": float(np.std(samples, ddof=1)),
            "latency_ms_median": float(np.median(samples)), "peak_vram_mb": float(peak),
            "rounds": rounds, "repeats_per_round": repeats}


def release(value):
    del value
    gc.collect(); torch.cuda.empty_cache()


def isolated_memory(dataset, seeds, configuration, batch_size, split, device):
    """Measure deployment peak with only the models needed by one configuration resident."""
    torch.cuda.empty_cache()
    if configuration in ("one full-coalition forward", "7-coalition backbone enumeration",
                         "3-coalition backbone enumeration"):
        values = [load_backbone_only(dataset, seeds[0], 0, device)]
        xs = features(split, batch_size, device)
        if configuration == "one full-coalition forward":
            function = lambda: full_forward(values[0], xs)
        else:
            function = lambda: coalition_forward(values[0], xs)
    elif configuration == "single-member complete RCG":
        values = [load_member(dataset, seeds[0], 0, device)]
        xs = features(split, batch_size, device)
        function = lambda: rcg_forward(values[0], xs)
    elif configuration == "five-member full ensemble":
        values = [load_backbone_only(dataset, seed, 0, device) for seed in seeds]
        xs = features(split, batch_size, device)
        function = lambda: torch.stack([full_forward(value, xs) for value in values]).mean(0)
    elif configuration == "five-member complete RCG":
        values = [load_member(dataset, seed, 0, device) for seed in seeds]
        xs = features(split, batch_size, device)
        function = lambda: torch.stack([rcg_forward(value, xs) for value in values]).mean(0)
    else:
        raise ValueError(configuration)
    torch.cuda.synchronize(device)
    resident = torch.cuda.memory_allocated(device)/1024**2
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        function()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)/1024**2
    result = {"resident_vram_mb": resident, "peak_vram_mb": peak,
              "activation_peak_mb": peak-resident}
    del function, xs, values
    gc.collect(); torch.cuda.empty_cache()
    return result


def features(split, batch_size, device):
    index = np.arange(batch_size) % len(split["y"])
    return [torch.as_tensor(x[index], dtype=torch.float32, device=device) for x in split["x"]]


def parameter_rows(member, dataset):
    rows = []
    for name, models in (("coalition backbone", [member["backbone"]]),
                         ("posterior ensemble (5)", member["posterior"]),
                         ("listwise router", [member["router"]]),
                         ("anchored mixer", [member["mixer"]])):
        total = sum(count_parameters(model)[0] for model in models)
        trainable = sum(count_parameters(model)[1] for model in models)
        rows.append({"dataset": dataset, "component": name, "parameters": total,
                     "trainable_parameters": trainable})
    total = sum(row["parameters"] for row in rows)
    rows.append({"dataset": dataset, "component": "single-member RCG total", "parameters": total,
                 "trainable_parameters": total})
    rows.append({"dataset": dataset, "component": "five-member RCG total", "parameters": total*5,
                 "trainable_parameters": total*5})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT/"results/e12_efficiency.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--memory-child", action="store_true")
    parser.add_argument("--memory-configuration")
    parser.add_argument("--memory-dataset", choices=("mosi", "cremad"))
    parser.add_argument("--memory-batch-size", type=int)
    parser.add_argument("--parameters-only", action="store_true")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E12 formal benchmark requires the configured CUDA GPU")
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    mosi, _ = load_dataset("mosi", ROOT/"data")
    cremad, _ = load_dataset("cremad", ROOT/"data")
    if args.memory_child:
        split = (mosi[0]["test"] if args.memory_dataset == "mosi" else cremad[0]["test"])
        value = isolated_memory(args.memory_dataset, SEEDS, args.memory_configuration,
                                args.memory_batch_size, split, device)
        print("E12_MEMORY_JSON="+json.dumps(value)); return
    if args.parameters_only:
        all_rows = []
        for dataset in ("mosi", "mosei", "cremad", "avmnist"):
            member = load_member(dataset, 11, 0, device)
            all_rows.extend(parameter_rows(member, dataset)); del member
            gc.collect(); torch.cuda.empty_cache()
        frame = pd.DataFrame(all_rows)
        Path(args.output).with_name("e12_parameters.csv").parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(Path(args.output).with_name("e12_parameters.csv"), index=False)
        print(frame.to_string(index=False)); return
    if args.memory_only:
        latency = pd.read_csv(args.output)
        memory_rows = []
        for row in latency.itertuples():
            command = [sys.executable, "-m", "scripts.run_e12_efficiency", "--memory-child",
                       "--memory-dataset", row.dataset, "--memory-configuration",
                       row.configuration, "--memory-batch-size", str(int(row.batch_size))]
            completed = subprocess.run(command, cwd=ROOT, check=True, text=True,
                                       capture_output=True)
            marker = [line for line in completed.stdout.splitlines()
                      if line.startswith("E12_MEMORY_JSON=")][-1]
            memory_rows.append(json.loads(marker.split("=", 1)[1]))
        for column in ("resident_vram_mb", "peak_vram_mb", "activation_peak_mb"):
            latency[column] = [row[column] for row in memory_rows]
        latency.to_csv(args.output, index=False)
        print(latency.to_string(index=False)); return

    members = [load_member("mosi", seed, 0, device) for seed in SEEDS]
    crema_member = load_member("cremad", 11, 0, device)

    latency = []
    for batch_size, repeats in ((1, 200), (64, 100)):
        mx = features(mosi[0]["test"], batch_size, device)
        cx = features(cremad[0]["test"], batch_size, device)
        latency.append({"dataset": "mosi", "batch_size": batch_size,
                        **benchmark("one full-coalition forward",
                                    lambda: full_forward(members[0], mx), device,
                                    args.rounds, repeats)})
        latency.append({"dataset": "mosi", "batch_size": batch_size,
                        **benchmark("7-coalition backbone enumeration",
                                    lambda: coalition_forward(members[0], mx), device,
                                    args.rounds, repeats)})
        latency.append({"dataset": "cremad", "batch_size": batch_size,
                        **benchmark("3-coalition backbone enumeration",
                                    lambda: coalition_forward(crema_member, cx), device,
                                    args.rounds, repeats)})
        latency.append({"dataset": "mosi", "batch_size": batch_size,
                        **benchmark("single-member complete RCG",
                                    lambda: rcg_forward(members[0], mx), device,
                                    args.rounds, repeats)})
        latency.append({"dataset": "mosi", "batch_size": batch_size,
                        **benchmark("five-member full ensemble",
                                    lambda: torch.stack([full_forward(m, mx) for m in members]).mean(0),
                                    device, args.rounds, max(20, repeats//5))})
        latency.append({"dataset": "mosi", "batch_size": batch_size,
                        **benchmark("five-member complete RCG",
                                    lambda: torch.stack([rcg_forward(m, mx) for m in members]).mean(0),
                                    device, args.rounds, max(20, repeats//5))})
    latency = pd.DataFrame(latency)
    base = latency[(latency.dataset == "mosi") &
                   (latency.configuration == "one full-coalition forward")]
    multiplier = dict(zip(base.batch_size, base.latency_ms_mean))
    latency["latency_multiplier_vs_full"] = [
        row.latency_ms_mean/multiplier[row.batch_size] for row in latency.itertuples()]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    latency.to_csv(output, index=False)
    pd.DataFrame(parameter_rows(members[0], "mosi")).to_csv(
        output.with_name("e12_parameters.csv"), index=False)
    metadata = {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                "cuda": torch.version.cuda, "warmup_iterations": 20,
                "timing": "7 wall-clock rounds with CUDA synchronization",
                "feature_encoders": "excluded; all methods use precomputed frozen features",
                "probability_aggregation": "included", "data_transfer": "excluded; tensors resident on GPU"}
    output.with_name("e12_benchmark_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(latency.to_string(index=False))
    print(pd.DataFrame(parameter_rows(members[0], "mosi")).to_string(index=False))


if __name__ == "__main__":
    main()
