"""Canonical definition and provenance helpers for the final RCG-Fusion system.

The final system aggregates the contribution-controlled A7 member actions with
selection-chosen simplex regularization, then falls back toward the equal
full-coalition ensemble.  Weights and the fallback coefficient are fitted on
the selection split only.  This module is the single
source of truth for the canonical final-system definition; subsequent formal
clean, stress, transfer, and case analyses must call it or consume its saved
predictions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .stable_ensemble import fit_simplex_weights, mix_actions, select_safe_shrinkage


METHOD_VERSION = "rcg-fusion-a8-v2"
METHOD_STAGES = (
    "grouped_oof_coalition_supervision",
    "posterior_analytic_contribution",
    "analytic_anchored_listwise_candidates",
    "anchored_candidate_mixer",
    "contribution_probability_shrinkage",
    "regularized_a7_member_convex_aggregation_with_full_fallback",
)


@dataclass(frozen=True)
class FinalSystemOutput:
    full_ensemble: np.ndarray
    a7_ensemble: np.ndarray
    convex_probability: np.ndarray
    final_probability: np.ndarray
    action_weights: np.ndarray
    fallback_rho: float
    aggregation_l2: float
    action_names: tuple[str, ...]
    fallback_grid: tuple[dict[str, float], ...]


def _member_actions(full_members: np.ndarray, a7_members: np.ndarray,
                    member_ids: Sequence[object] | None = None):
    full = np.asarray(full_members, dtype=np.float64)
    a7 = np.asarray(a7_members, dtype=np.float64)
    if full.ndim != 3 or full.shape != a7.shape:
        raise ValueError("full and A7 members must have identical [E,N,K] shapes")
    if member_ids is None:
        member_ids = tuple(range(full.shape[0]))
    if len(member_ids) != full.shape[0]:
        raise ValueError("one member id is required per ensemble member")
    actions = np.empty((2*full.shape[0], full.shape[1], full.shape[2]), dtype=np.float64)
    names: list[str] = []
    for index, member_id in enumerate(member_ids):
        actions[2*index] = full[index]
        actions[2*index+1] = a7[index]
        names.extend((f"member_{member_id}_full", f"member_{member_id}_a7"))
    return actions, tuple(names)


def fit_final_system(selection_full_members: np.ndarray,
                     selection_a7_members: np.ndarray,
                     selection_labels: np.ndarray,
                     test_full_members: np.ndarray,
                     test_a7_members: np.ndarray,
                     *, member_ids: Sequence[object] | None = None,
                     simplex_l2_grid: Iterable[float] = (1e-3, 1e-2, 1e-1, 1., 10.),
                     regularization_tolerance: float = 1e-3,
                     fallback_grid: Iterable[float] | None = None) -> FinalSystemOutput:
    """Fit the canonical A8 aggregation on selection and apply it to test.

    Test labels are deliberately absent from this interface.
    """
    selection_full_members = np.asarray(selection_full_members, dtype=np.float64)
    selection_actions = np.asarray(selection_a7_members, dtype=np.float64)
    test_actions = np.asarray(test_a7_members, dtype=np.float64)
    if selection_full_members.ndim != 3 or selection_actions.shape != selection_full_members.shape:
        raise ValueError("full and A7 members must have identical [E,N,K] shapes")
    if test_actions.shape != np.asarray(test_full_members).shape:
        raise ValueError("test full and A7 members must have identical [E,N,K] shapes")
    if member_ids is None:
        member_ids = tuple(range(selection_actions.shape[0]))
    if len(member_ids) != selection_actions.shape[0]:
        raise ValueError("one member id is required per ensemble member")
    action_names = tuple(f"member_{member_id}_a7" for member_id in member_ids)
    labels = np.asarray(selection_labels, dtype=np.int64)
    if selection_actions.shape[1] != len(labels):
        raise ValueError("selection labels are not aligned with member actions")
    candidates = []
    for l2 in tuple(float(value) for value in simplex_l2_grid):
        if l2 <= 0:
            raise ValueError("simplex regularization values must be positive")
        candidate_weights = fit_simplex_weights(selection_actions, labels, l2=l2)
        candidate_probability = mix_actions(selection_actions, candidate_weights)
        candidate_nll = float(-np.log(np.clip(candidate_probability[
            np.arange(len(labels)), labels], 1e-12, 1)).mean())
        candidates.append((l2, candidate_nll, candidate_weights))
    best_nll = min(value[1] for value in candidates)
    eligible = [value for value in candidates
                if value[1] <= best_nll+float(regularization_tolerance)]
    chosen_l2, _, weights = max(eligible, key=lambda value: value[0])
    selection_convex = mix_actions(selection_actions, weights)
    test_convex = mix_actions(test_actions, weights)
    selection_full = np.asarray(selection_full_members, dtype=np.float64).mean(0)
    test_full = np.asarray(test_full_members, dtype=np.float64).mean(0)
    test_a7 = np.asarray(test_a7_members, dtype=np.float64).mean(0)
    grid = None if fallback_grid is None else np.asarray(tuple(fallback_grid), dtype=float)
    rho, records = select_safe_shrinkage(selection_full, selection_convex, labels, grid=grid)
    final = (1-rho)*test_full+rho*test_convex
    return FinalSystemOutput(
        full_ensemble=test_full,
        a7_ensemble=test_a7,
        convex_probability=test_convex,
        final_probability=final,
        action_weights=weights,
        fallback_rho=float(rho),
        aggregation_l2=float(chosen_l2),
        action_names=action_names,
        fallback_grid=tuple({key: float(value) for key, value in row.items()} for row in records),
    )


def prediction_hash(sample_ids: Sequence[object], folds: Sequence[int],
                    probabilities: np.ndarray, *,
                    method_version: str = METHOD_VERSION) -> str:
    """Return a stable SHA-256 hash for ordered final predictions."""
    ids = np.asarray(sample_ids, dtype=str)
    fold = np.asarray(folds, dtype=np.int64)
    probability = np.asarray(probabilities, dtype="<f8")
    if probability.ndim != 2 or len(ids) != len(probability) or len(fold) != len(probability):
        raise ValueError("sample ids, folds, and [N,K] probabilities must align")
    order = np.lexsort((ids, fold))
    digest = hashlib.sha256()
    digest.update(method_version.encode("utf-8"))
    digest.update(json.dumps({"n": len(ids), "classes": probability.shape[1]},
                             sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for index in order:
        encoded = ids[index].encode("utf-8")
        digest.update(int(fold[index]).to_bytes(4, "little", signed=True))
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(np.ascontiguousarray(probability[index]).tobytes())
    return digest.hexdigest()


def method_manifest() -> dict[str, object]:
    return {
        "method_version": METHOD_VERSION,
        "stages": list(METHOD_STAGES),
        "selection_only_parameters": ["A7 strength", "A7 simplex regularization",
                                      "A7 simplex weights", "fallback rho"],
        "test_labels_role": "evaluation and offline analysis only",
    }
