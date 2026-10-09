"""I3-3: canonical A7 member aggregation and full-ensemble fallback audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from rcg.anchored_mixer import AnchoredCandidateMixer
from rcg.anchored_mixer_pipeline import predict_mixer
from rcg.final_system import METHOD_VERSION, fit_final_system, prediction_hash
from rcg.listwise_router import AnalyticResidualListwiseRouter
from rcg.listwise_router_pipeline import apply_residual_blend, predict_router
from rcg.posterior_analytic_pipeline import _load_backbone
from rcg.posterior_shrinkage_pipeline import probability_metrics
from rcg.projected_fusion_pipeline import load_posteriors
from rcg.rcg_fusion import nonempty_coalitions
from rcg.rcg_fusion_pipeline import attach_observed_losses, load_dataset, predict_outputs
from rcg.stable_ensemble import fit_simplex_weights, mix_actions, select_safe_shrinkage

from run_e4_anchored_routing import CONFIGS, SEEDS
from run_e5_candidate_mixer_ablation import make_actions
from run_e6_adaptive_shrinkage import gate_values, shrink


ROOT = Path(__file__).resolve().parents[1]


def normalize(probability):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, None)
    return probability/probability.sum(1, keepdims=True)


def fit_temperature(probability, labels):
    logits = np.log(np.clip(probability, 1e-8, 1))
    labels = np.asarray(labels, int)
    def objective(log_temperature):
        scaled = logits/np.exp(log_temperature)
        return float((logsumexp(scaled, axis=1)-scaled[np.arange(len(labels)), labels]).mean())
    return float(np.exp(minimize_scalar(objective, bounds=(-4, 4), method="bounded").x))


def temperature_scale(probability, temperature):
    logits = np.log(np.clip(probability, 1e-8, 1))/temperature
    logits -= logits.max(1, keepdims=True)
    return normalize(np.exp(logits))


def logit_features(actions):
    return np.concatenate([np.log(np.clip(action, 1e-8, 1)) for action in actions], axis=1)


def fit_logit_stacking(actions, labels):
    model = LogisticRegression(C=1., max_iter=2000, solver="lbfgs")
    model.fit(logit_features(actions), labels)
    return model


def evaluate(probability, labels, full):
    metric, prediction = probability_metrics(probability, labels)
    full_metric, full_prediction = probability_metrics(full, labels)
    row = np.arange(len(labels))
    loss = -np.log(np.clip(probability[row, labels], 1e-12, 1))
    full_loss = -np.log(np.clip(full[row, labels], 1e-12, 1))
    harm = np.maximum(loss-full_loss, 0)
    return {**metric,
            "accuracy_gain": metric["accuracy"]-full_metric["accuracy"],
            "nll_gain": full_metric["nll"]-metric["nll"],
            "relative_error_reduction": ((metric["accuracy"]-full_metric["accuracy"])/
                                         max(1-full_metric["accuracy"], 1e-12)),
            "relative_nll_reduction": ((full_metric["nll"]-metric["nll"])/
                                       max(full_metric["nll"], 1e-12)),
            "negative_flip_rate": float(((full_prediction == labels) &
                                          (prediction != labels)).mean()),
            "correction_rate": float(((full_prediction != labels) &
                                       (prediction == labels)).mean()),
            "clipped_harm": float(np.minimum(harm, 1).mean()),
            "harm_p95": float(np.quantile(harm, .95))}


def selected_strength(dataset, fold, seed):
    frame = pd.read_csv(ROOT/f"runs/formal-e6-{dataset}/metrics_by_fold_seed.csv")
    row = frame[(frame.variant == "analytic_pi_g2") &
                (frame.fold == fold) & (frame.train_seed == seed)]
    if len(row) != 1:
        raise ValueError(f"missing canonical A7 strength: {dataset} fold={fold} seed={seed}")
    return float(row.iloc[0].strength)


def member_output(dataset, fold, splits, names, seed, device):
    cfg = CONFIGS[dataset]; masks = nonempty_coalitions(len(names))
    top_k = 3 if len(names) >= 3 else 2
    backbone, temperatures, dims, classes = _load_backbone(
        cfg["base"]/f"fold_{fold}/seed_{seed}/backbone.pt", device)
    checkpoint = ROOT/f"runs/formal-e5-{dataset}/fold_{fold}/seed_{seed}"
    router_saved = torch.load(checkpoint/"router.pt", map_location=device, weights_only=True)
    router = AnalyticResidualListwiseRouter(dims, classes, len(masks), residual_scale=.5).to(device)
    router.load_state_dict(router_saved["state_dict"]); router.eval()
    mixer_saved = torch.load(checkpoint/"anchored_harm_oracle.pt",
                             map_location=device, weights_only=True)
    mixer = AnchoredCandidateMixer(classes, anchor_full=True).to(device)
    mixer.load_state_dict(mixer_saved["state_dict"]); mixer.eval()
    strength = selected_strength(dataset, fold, seed); result = {}
    for split_name in ("selection", "test"):
        raw = predict_outputs(backbone, splits[split_name]["x"], device, temperatures)
        bundle = attach_observed_losses(raw, splits[split_name]["y"])
        _, posterior = load_posteriors(cfg["posterior"], seed, dims, classes, masks,
                                       splits[split_name], bundle["probabilities"], device,
                                       fold_index=fold)
        route = predict_router(router, splits[split_name], bundle["probabilities"],
                               masks, device, posterior)
        route = apply_residual_blend(route, float(router_saved["blend"]))
        actions, _ = make_actions(bundle["probabilities"], posterior, route, top_k)
        mixture = predict_mixer(mixer, actions, posterior, device)["probability"]
        gate = gate_values("analytic_pi_g2", actions[:, 0], mixture, posterior)
        a7, _ = shrink(actions[:, 0], mixture, gate, strength)
        result[split_name] = {"full": actions[:, 0], "a7": a7, "posterior": posterior}
    return result


def aggregate_folds(frame):
    rows = []
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns
               if column not in {"fold", "n_test"}]
    for method, values in frame.groupby("method"):
        weights = values.n_test.to_numpy(float)
        row = {"method": method, "n_test": int(weights.sum())}
        row.update({column: float(np.average(values[column], weights=weights))
                    for column in numeric})
        rows.append(row)
    return pd.DataFrame(rows)


def run_dataset(dataset, device):
    folds, names = load_dataset(dataset, ROOT/"data")
    root = ROOT/f"runs/formal-i3-3-{dataset}"; root.mkdir(parents=True, exist_ok=True)
    metric_rows, predictions, weights, fallbacks, hashes = [], [], [], [], []
    for fold, splits in enumerate(folds):
        members = {split: {kind: [] for kind in ("full", "a7", "posterior")}
                   for split in ("selection", "test")}
        for seed in SEEDS:
            output = member_output(dataset, fold, splits, names, seed, device)
            for split in members:
                for kind in members[split]:
                    members[split][kind].append(output[split][kind])
        for split in members:
            for kind in members[split]:
                members[split][kind] = np.asarray(members[split][kind])
        selection_y, test_y = splits["selection"]["y"], splits["test"]["y"]
        sel_full = members["selection"]["full"].mean(0)
        test_full = members["test"]["full"].mean(0)
        sel_a7 = members["selection"]["a7"].mean(0)
        test_a7 = members["test"]["a7"].mean(0)
        sel_posterior = members["selection"]["posterior"].mean(0)
        test_posterior = members["test"]["posterior"].mean(0)
        selection_actions = np.empty((2*len(SEEDS), *sel_full.shape))
        test_actions = np.empty((2*len(SEEDS), *test_full.shape))
        for index in range(len(SEEDS)):
            selection_actions[2*index:2*index+2] = [members["selection"]["full"][index],
                                                     members["selection"]["a7"][index]]
            test_actions[2*index:2*index+2] = [members["test"]["full"][index],
                                               members["test"]["a7"][index]]
        temperature = fit_temperature(sel_full, selection_y)
        sel_temperature = temperature_scale(sel_full, temperature)
        test_temperature = temperature_scale(test_full, temperature)
        a7_temperature = fit_temperature(sel_a7, selection_y)
        sel_a7_temperature = temperature_scale(sel_a7, a7_temperature)
        test_a7_temperature = temperature_scale(test_a7, a7_temperature)
        logit = fit_logit_stacking(selection_actions, selection_y)
        sel_logit = normalize(logit.predict_proba(logit_features(selection_actions)))
        test_logit = normalize(logit.predict_proba(logit_features(test_actions)))
        simplex = fit_simplex_weights(selection_actions, selection_y, l2=1e-3)
        sel_convex = mix_actions(selection_actions, simplex)
        test_convex = mix_actions(test_actions, simplex)
        rho, grid = select_safe_shrinkage(sel_full, sel_convex, selection_y)
        sel_safe = normalize((1-rho)*sel_full+rho*sel_convex)
        test_safe = normalize((1-rho)*test_full+rho*test_convex)
        # Hierarchical aggregation first removes member-level randomness, then
        # fits only two stable ensemble actions.  This is the revised stable
        # model-level path; the ten-action simplex remains an overfitting audit.
        ensemble_selection_actions = np.asarray([sel_full, sel_a7])
        ensemble_test_actions = np.asarray([test_full, test_a7])
        hierarchical_weights = fit_simplex_weights(
            ensemble_selection_actions, selection_y, l2=1e-3)
        sel_hierarchical = mix_actions(ensemble_selection_actions, hierarchical_weights)
        test_hierarchical = mix_actions(ensemble_test_actions, hierarchical_weights)
        hierarchical_rho, hierarchical_grid = select_safe_shrinkage(
            sel_full, sel_hierarchical, selection_y)
        sel_hierarchical_safe = normalize(
            (1-hierarchical_rho)*sel_full+hierarchical_rho*sel_hierarchical)
        test_hierarchical_safe = normalize(
            (1-hierarchical_rho)*test_full+hierarchical_rho*test_hierarchical)
        a7_weights = fit_simplex_weights(members["selection"]["a7"], selection_y, l2=1e-3)
        sel_a7_convex = mix_actions(members["selection"]["a7"], a7_weights)
        test_a7_convex = mix_actions(members["test"]["a7"], a7_weights)
        a7_rho, a7_grid = select_safe_shrinkage(sel_full, sel_a7_convex, selection_y)
        sel_a7_safe = normalize((1-a7_rho)*sel_full+a7_rho*sel_a7_convex)
        test_a7_safe = normalize((1-a7_rho)*test_full+a7_rho*test_a7_convex)
        regularized_a7 = {}
        regularization_records = []
        for l2 in (1e-3, 1e-2, 1e-1, 1., 10.):
            candidate_weight = fit_simplex_weights(
                members["selection"]["a7"], selection_y, l2=l2)
            candidate_selection = mix_actions(members["selection"]["a7"], candidate_weight)
            candidate_test = mix_actions(members["test"]["a7"], candidate_weight)
            candidate_nll = float(-np.log(np.clip(candidate_selection[
                np.arange(len(selection_y)), selection_y], 1e-12, 1)).mean())
            regularized_a7[l2] = (candidate_selection, candidate_test, candidate_weight)
            regularization_records.append({"l2": l2, "selection_nll": candidate_nll})
        best_regularization_nll = min(row["selection_nll"] for row in regularization_records)
        # One-tolerance rule: prefer the strongest shrinkage whose selection NLL
        # is within 1e-3 of the empirical optimum.
        chosen_l2 = max(row["l2"] for row in regularization_records
                        if row["selection_nll"] <= best_regularization_nll+1e-3)
        sel_a7_regularized, test_a7_regularized, regularized_weight = regularized_a7[chosen_l2]
        regularized_rho, regularized_grid = select_safe_shrinkage(
            sel_full, sel_a7_regularized, selection_y)
        sel_a7_regularized_safe = normalize(
            (1-regularized_rho)*sel_full+regularized_rho*sel_a7_regularized)
        test_a7_regularized_safe = normalize(
            (1-regularized_rho)*test_full+regularized_rho*test_a7_regularized)
        canonical = fit_final_system(
            members["selection"]["full"], members["selection"]["a7"], selection_y,
            members["test"]["full"], members["test"]["a7"], member_ids=SEEDS,
            simplex_l2_grid=(1e-3, 1e-2, 1e-1, 1., 10.))
        if not (np.allclose(canonical.convex_probability, test_a7_regularized) and
                np.allclose(canonical.final_probability, test_a7_regularized_safe) and
                np.isclose(canonical.fallback_rho, regularized_rho) and
                np.isclose(canonical.aggregation_l2, chosen_l2)):
            raise AssertionError("I3-3 aggregation differs from canonical P0 system")
        methods = {
            "full_ensemble": (sel_full, test_full),
            "temperature_full_ensemble": (sel_temperature, test_temperature),
            "a7_equal_ensemble": (sel_a7, test_a7),
            "temperature_a7_ensemble": (sel_a7_temperature, test_a7_temperature),
            "posterior_ensemble": (sel_posterior, test_posterior),
            "all_actions_equal": (selection_actions.mean(0), test_actions.mean(0)),
            "logit_stacking": (sel_logit, test_logit),
            "simplex_convex": (sel_convex, test_convex),
            "simplex_safe_fallback": (sel_safe, test_safe),
            "hierarchical_convex": (sel_hierarchical, test_hierarchical),
            "hierarchical_safe_fallback": (sel_hierarchical_safe, test_hierarchical_safe),
            "a7_member_convex": (sel_a7_convex, test_a7_convex),
            "a7_member_safe_fallback": (sel_a7_safe, test_a7_safe),
            "regularized_a7_safe_fallback": (sel_a7_regularized_safe,
                                               test_a7_regularized_safe),
        }
        for method, (selection_probability, test_probability) in methods.items():
            metric_rows.append({"dataset": dataset, "fold": fold, "method": method,
                                "n_test": len(test_y), "temperature": temperature,
                                "rho": rho,
                                "selection_accuracy": float((selection_probability.argmax(1) == selection_y).mean()),
                                "selection_nll": float(-np.log(np.clip(selection_probability[
                                    np.arange(len(selection_y)), selection_y], 1e-12, 1)).mean()),
                                **evaluate(test_probability, test_y, test_full)})
            frame = pd.DataFrame({"dataset": dataset, "fold": fold, "method": method,
                                  "method_version": METHOD_VERSION,
                                  "sample_id": splits["test"]["id"],
                                  "group_or_video_id": splits["test"]["group"],
                                  "label": test_y})
            for cls in range(test_probability.shape[1]):
                frame[f"p{cls}"] = test_probability[:, cls]
            predictions.append(frame)
        for name, value in zip(canonical.action_names, canonical.action_weights):
            weights.append({"dataset": dataset, "fold": fold, "action": name,
                            "weight": value, "rho": rho})
        for name, value in zip(("full_ensemble", "a7_ensemble"), hierarchical_weights):
            weights.append({"dataset": dataset, "fold": fold,
                            "action": f"hierarchical_{name}", "weight": value,
                            "rho": hierarchical_rho})
        for seed, value in zip(SEEDS, a7_weights):
            weights.append({"dataset": dataset, "fold": fold,
                            "action": f"a7_member_{seed}", "weight": value,
                            "rho": a7_rho})
        for seed, value in zip(SEEDS, regularized_weight):
            weights.append({"dataset": dataset, "fold": fold,
                            "action": f"regularized_a7_member_{seed}", "weight": value,
                            "rho": regularized_rho})
        fallbacks.append({"dataset": dataset, "fold": fold, "rho": rho,
                          "hierarchical_rho": hierarchical_rho,
                          "temperature": temperature, "a7_temperature": a7_temperature,
                          "selection_grid": json.dumps(grid),
                          "hierarchical_selection_grid": json.dumps(hierarchical_grid)})
        fallbacks[-1].update({"a7_member_rho": a7_rho,
                              "a7_member_selection_grid": json.dumps(a7_grid),
                              "regularized_a7_l2": chosen_l2,
                              "regularized_a7_rho": regularized_rho,
                              "regularized_a7_selection_grid": json.dumps(regularized_grid),
                              "regularization_records": json.dumps(regularization_records)})
        digest = prediction_hash(splits["test"]["id"], np.full(len(test_y), fold),
                                 canonical.final_probability, method_version=METHOD_VERSION)
        hashes.append({"dataset": dataset, "fold": fold, "prediction_hash": digest})
        print(f"{dataset} fold={fold}: rho={rho:.1f} full={evaluate(test_full,test_y,test_full)['nll']:.4f} "
              f"A7={evaluate(test_a7,test_y,test_full)['nll']:.4f} "
              f"convex={evaluate(test_convex,test_y,test_full)['nll']:.4f} "
              f"safe={evaluate(test_safe,test_y,test_full)['nll']:.4f} "
              f"hier={evaluate(test_hierarchical_safe,test_y,test_full)['nll']:.4f} "
              f"a7simplex={evaluate(test_a7_safe,test_y,test_full)['nll']:.4f} "
              f"regularized={evaluate(test_a7_regularized_safe,test_y,test_full)['nll']:.4f}",
              flush=True)
    metrics = pd.DataFrame(metric_rows); metrics.to_csv(root/"metrics_by_fold.csv", index=False)
    aggregate_folds(metrics).to_csv(root/"metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(root/"predictions.parquet", index=False)
    pd.DataFrame(weights).to_csv(root/"weights.csv", index=False)
    pd.DataFrame(fallbacks).to_csv(root/"fallbacks.csv", index=False)
    pd.DataFrame(hashes).to_csv(root/"prediction_hashes.csv", index=False)
    (root/"manifest.json").write_text(json.dumps({
        "experiment": "I3-3 canonical A7 aggregation and fallback",
        "dataset": dataset, "method_version": METHOD_VERSION, "seeds": list(SEEDS),
        "selection_role": "fit temperature, stacking, simplex weights, and fallback rho",
        "test_labels_role": "evaluation only",
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*CONFIGS, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    for dataset in (CONFIGS if args.dataset == "all" else (args.dataset,)):
        run_dataset(dataset, device)


if __name__ == "__main__":
    main()
