"""E9 clean-trained continuous feature degradation evaluation.

All member checkpoints and selection-chosen policies are frozen.  A corruption
field is shared across methods, training seeds, and nested severity levels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from rcg.anchored_mixer import AnchoredCandidateMixer
from rcg.anchored_mixer_pipeline import analytic_action_probability, predict_mixer
from rcg.io import load_json
from rcg.listwise_router import AnalyticResidualListwiseRouter
from rcg.listwise_router_pipeline import apply_residual_blend, predict_router
from rcg.posterior_analytic import RelationalPosteriorEstimator
from rcg.posterior_analytic_pipeline import (_load_backbone, calibrate_members,
                                                predict_posterior)
from rcg.posterior_shrinkage_pipeline import probability_metrics
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import (load_dataset, predict_outputs,
                                     stress_conditions)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from run_e4_anchored_routing import CONFIGS, SEEDS  # noqa: E402
from run_e5_candidate_mixer_ablation import make_actions  # noqa: E402
from run_e6_adaptive_shrinkage import gate_values, shrink  # noqa: E402


def policy_strengths(dataset, fold, seed):
    frame = pd.read_csv(ROOT/f"runs/formal-e6-{dataset}/metrics_by_fold_seed.csv")
    frame = frame[(frame.fold == fold) & (frame.train_seed == seed)]
    return {variant: float(frame.loc[frame.variant == variant, "strength"].iloc[0])
            for variant in ("max_confidence", "negative_entropy", "analytic_pi_g2")}


def load_context(dataset, fold, seed, masks, device, backbone_path=None):
    cfg = CONFIGS[dataset]
    backbone, temperatures, dims, classes = _load_backbone(
        (Path(backbone_path) if backbone_path is not None else
         cfg["base"]/f"fold_{fold}/seed_{seed}/backbone.pt"), device)
    posterior_root = Path(cfg["posterior"])/f"fold_{fold}/seed_{seed}"
    posterior_members = []
    for path in sorted(posterior_root.glob("posterior_*.pt")):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = RelationalPosteriorEstimator(
            dims, classes, len(masks), use_relations=checkpoint["use_relations"]).to(device)
        model.load_state_dict(checkpoint["state_dict"]); model.eval(); posterior_members.append(model)
    posterior_temperature = load_json(posterior_root/"metrics.json")["posterior_temperature"]
    e5 = ROOT/f"runs/formal-e5-{dataset}/fold_{fold}/seed_{seed}"
    router_saved = torch.load(e5/"router.pt", map_location=device, weights_only=True)
    router = AnalyticResidualListwiseRouter(dims, classes, len(masks), residual_scale=.5).to(device)
    router.load_state_dict(router_saved["state_dict"]); router.eval()
    mixer_saved = torch.load(e5/"anchored_harm_oracle.pt", map_location=device, weights_only=True)
    mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
    mixer.load_state_dict(mixer_saved["state_dict"]); mixer.eval()
    return {"seed": seed, "backbone": backbone, "temperatures": temperatures,
            "posterior_members": posterior_members, "posterior_temperature": posterior_temperature,
            "router": router, "router_blend": float(router_saved["blend"]), "mixer": mixer,
            "strengths": policy_strengths(dataset, fold, seed), "classes": classes}


def posterior(context, split, probabilities, masks, device, availability=None):
    members = [predict_posterior(
        model, split, probabilities, masks, device, availability=availability)["posterior"]
               for model in context["posterior_members"]]
    return np.mean(calibrate_members(members, context["posterior_temperature"]), axis=0)


def member_predict(context, split, masks, top_k, device):
    outputs = predict_outputs(context["backbone"], split["x"], device, context["temperatures"])
    probabilities = outputs["probabilities"]
    q = posterior(context, split, probabilities, masks, device)
    route = predict_router(context["router"], split, probabilities, masks, device, q)
    route = apply_residual_blend(route, context["router_blend"])
    actions, _ = make_actions(probabilities, q, route, top_k)
    mixer = predict_mixer(context["mixer"], actions, q, device)
    full, mixture = actions[:, 0], mixer["probability"]
    result = {"full_ensemble": full, "posterior_ensemble": q, "raw_mixer_ensemble": mixture}
    for controller, name in [("max_confidence", "confidence_shrink_ensemble"),
                             ("negative_entropy", "entropy_shrink_ensemble"),
                             ("analytic_pi_g2", "A7_equal_ensemble")]:
        gate = gate_values(controller, full, mixture, q)
        result[name], alpha = shrink(full, mixture, gate, context["strengths"][controller])
        if controller == "analytic_pi_g2": result["a7_alpha"] = alpha
    result["analytic_pi"] = analytic_action_probability(full, mixture, q, tolerance=.01)
    result["mixer_weight"] = mixer["weight"]
    return result


def metadata(condition):
    if condition == "clean": return "clean", "none", 0., -1
    kind, modality, level, seed = condition.split(":")
    return kind, modality, float(level), int(seed.removeprefix("seed"))


def metrics(probability, labels, full):
    value, prediction = probability_metrics(probability, labels)
    base, base_prediction = probability_metrics(full, labels)
    rows = np.arange(len(labels)); loss = -np.log(np.clip(probability[rows, labels], 1e-12, 1))
    full_loss = -np.log(np.clip(full[rows, labels], 1e-12, 1))
    return {**value, "accuracy_gain": value["accuracy"]-base["accuracy"],
            "nll_gain": base["nll"]-value["nll"],
            "negative_flip_rate": float(((base_prediction == labels)&(prediction != labels)).mean()),
            "correction_rate": float(((base_prediction != labels)&(prediction == labels)).mean()),
            "clipped_harm": float(np.minimum(np.maximum(loss-full_loss, 0), 1).mean())}


def run_dataset(dataset, device):
    folds, names = load_dataset(dataset, ROOT/"data"); masks = nonempty_coalitions(len(names))
    top_k = 3 if len(names) >= 3 else 2; root = ROOT/f"runs/formal-e9-clean-{dataset}"
    root.mkdir(parents=True, exist_ok=True); metric_rows=[]; prediction_frames=[]
    for fold, splits in enumerate(folds):
        contexts=[load_context(dataset, fold, seed, masks, device) for seed in SEEDS]
        e8_weights=pd.read_csv(ROOT/f"runs/formal-e8-{dataset}/weights.csv")
        e8_weights=e8_weights[e8_weights.fold==fold].weight.to_numpy(float)
        rho=float(pd.read_csv(ROOT/f"runs/formal-e8-{dataset}/metrics_by_fold.csv").query(
            "fold == @fold and method == 'A8_safe_fallback'").rho.iloc[0])
        for condition, changed, _ in stress_conditions(splits["test"], names):
            if condition.startswith("missing:"): continue
            outputs=[member_predict(context, changed, masks, top_k, device) for context in contexts]
            method_names=["full_ensemble","posterior_ensemble","raw_mixer_ensemble",
                          "confidence_shrink_ensemble","entropy_shrink_ensemble","A7_equal_ensemble"]
            ensemble={method:np.mean([x[method] for x in outputs],axis=0) for method in method_names}
            # P0 v2 fits the simplex only over the five contribution-controlled
            # A7 member actions.  The former v1 chain interleaved full and A7
            # actions and must not be used in normative stress evaluation.
            a7_members=np.asarray([output["A7_equal_ensemble"] for output in outputs])
            if len(e8_weights) != len(a7_members):
                raise ValueError("P0 v2 A8 weights must align with A7 members")
            convex=np.tensordot(e8_weights,a7_members,axes=(0,0))
            ensemble["A8_safe_fallback"]=(1-rho)*ensemble["full_ensemble"]+rho*convex
            kind,modality,level,corruption_seed=metadata(condition);labels=changed["y"]
            alpha=float(np.mean([x["a7_alpha"].mean() for x in outputs]))
            weights=np.concatenate([x["mixer_weight"] for x in outputs])
            pi=np.mean([x["analytic_pi"] for x in outputs],axis=0)
            mix=np.mean([x["raw_mixer_ensemble"] for x in outputs],axis=0)
            rows=np.arange(len(labels)); realized=(
                np.log(np.clip(mix[rows,labels],1e-12,1)/np.clip(ensemble["full_ensemble"][rows,labels],1e-12,1))>.01)
            auroc=float(roc_auc_score(realized,pi)) if np.unique(realized).size==2 else np.nan
            for method, probability in ensemble.items():
                metric_rows.append({"dataset":dataset,"fold":fold,"method":method,"condition":condition,
                                    "corruption_type":kind,"affected_modality":modality,
                                    "corruption_level":level,"corruption_seed":corruption_seed,
                                    "n_test":len(labels),"mean_a7_alpha":alpha,
                                    "mean_full_weight":float(weights[:,0].mean()),
                                    "mean_posterior_weight":float(weights[:,1].mean()),
                                    "contribution_auroc":auroc,
                                    **metrics(probability,labels,ensemble["full_ensemble"])})
                if method in ("full_ensemble","A7_equal_ensemble","A8_safe_fallback"):
                    frame=pd.DataFrame({"dataset":dataset,"fold":fold,"method":method,
                                        "condition":condition,"corruption_type":kind,
                                        "affected_modality":modality,"corruption_level":level,
                                        "corruption_seed":corruption_seed,"sample_id":changed["id"],
                                        "group_or_video_id":changed["group"],"label":labels})
                    for k in range(probability.shape[1]):frame[f"p{k}"]=probability[:,k]
                    prediction_frames.append(frame)
            print(f"{dataset} fold={fold} {condition}: A0={metric_rows[-7]['nll']:.3f} "
                  f"A7={metric_rows[-2]['nll']:.3f} A8={metric_rows[-1]['nll']:.3f}",flush=True)
        del contexts
        if device=="cuda": torch.cuda.empty_cache()
    pd.DataFrame(metric_rows).to_csv(root/"metrics_by_condition_fold.csv",index=False)
    pd.concat(prediction_frames,ignore_index=True).to_parquet(root/"predictions.parquet",index=False)
    (root/"manifest.json").write_text(json.dumps({"experiment":"E9 clean-trained continuous degradation",
        "dataset":dataset,"method_version":"rcg-fusion-a8-v2",
        "noise_levels":[.25,.5,1.,2.],"mask_levels":[.25,.5,.75],
        "corruption_seeds":[101,202,303],"shared_corruption_across_methods_and_members":True,
        "training_protocol":"clean-only","test_labels_role":"evaluation only"},indent=2),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",choices=(*CONFIGS,"all"),default="all")
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto");args=parser.parse_args()
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else args.device;device="cpu" if device=="auto" else device
    for dataset in (CONFIGS if args.dataset=="all" else (args.dataset,)):run_dataset(dataset,device)


if __name__=="__main__":main()
