import numpy as np
import torch

from rcg.listwise_router import (AnalyticResidualListwiseRouter,
                                 anchor_preserving_candidates, candidate_recall,
                                 build_soft_coalition_targets,
                                 listwise_router_objective, ranking_metrics)
from rcg.rcg_fusion import nonempty_coalitions
from rcg.listwise_router_pipeline import (apply_residual_blend,
                                           select_residual_blend)


def test_soft_targets_are_normalized_and_pairwise_oriented():
    losses = np.asarray([[[.2, .4, .3], [.5, .1, .2]],
                         [[.1, .5, .3], [.4, .2, .3]]])
    target = build_soft_coalition_targets(losses, oracle_temperature=.1)
    np.testing.assert_allclose(target["soft_oracle"].sum(1), 1, atol=1e-6)
    assert target["pairwise_preference"][0, 0, 1] == 1
    assert target["pairwise_preference"][0, 1, 0] == 0
    assert target["oracle_coalition_index"].tolist() == [0, 1]
    assert np.all((target["agreement_weight"] >= 0)
                  & (target["agreement_weight"] <= 1))


def test_router_starts_from_analytic_prior_and_objective_is_finite():
    torch.manual_seed(3); masks = torch.tensor(nonempty_coalitions(2), dtype=torch.float32)
    model = AnalyticResidualListwiseRouter([4, 5], 3, len(masks), dropout=0)
    xs = [torch.randn(6, 4), torch.randn(6, 5)]
    probability = torch.softmax(torch.randn(6, len(masks), 3), -1)
    output = model(xs, probability, masks)
    torch.testing.assert_close(output["score"],
                               output["analytic_score"]-output["analytic_score"][:, -1, None])
    synthetic = np.random.default_rng(4).uniform(.1, 1.2, size=(2, 6, len(masks)))
    raw = build_soft_coalition_targets(synthetic)
    targets = {key: torch.as_tensor(raw[key]) for key in
               ("soft_oracle", "pairwise_preference", "stable_improvement", "agreement_weight")}
    objective, parts = listwise_router_objective(
        output, torch.randint(0, 3, (6,)), targets)
    assert torch.isfinite(objective)
    assert all(torch.isfinite(value) for value in parts.values())

    pair_only, _ = listwise_router_objective(
        output, torch.randint(0, 3, (6,)), targets,
        list_weight=0, pair_weight=1, stable_weight=0)
    assert torch.isfinite(pair_only)


def test_coalition_query_permutation_is_equivariant():
    torch.manual_seed(5); masks = torch.tensor(nonempty_coalitions(3), dtype=torch.float32)
    model = AnalyticResidualListwiseRouter([3, 4, 5], 2, len(masks), dropout=0).eval()
    xs = [torch.randn(4, 3), torch.randn(4, 4), torch.randn(4, 5)]
    probability = torch.softmax(torch.randn(4, len(masks), 2), -1)
    permutation = torch.tensor([3, 0, 5, 1, 6, 2, 4])
    with torch.inference_mode():
        first = model(xs, probability, masks)["score"]
        second = model(xs, probability[:, permutation], masks[permutation])["score"]
    inverse = torch.argsort(permutation)
    assert permutation[-1] != len(masks)-1  # exercise arbitrary query order
    torch.testing.assert_close(second[:, inverse], first, atol=1e-5, rtol=1e-5)


def test_ranking_metrics_recognize_perfect_order():
    losses = np.asarray([[.3, .1, .2], [.2, .4, .1]])
    result = ranking_metrics(-losses, losses)
    assert result["top1"] == 1 and result["top2"] == 1
    assert result["pairwise_accuracy"] == 1
    assert abs(result["ndcg"]-1) < 1e-12


def test_residual_blend_can_fall_back_to_analytic_prior():
    output = {"analytic_score": np.asarray([[2., 1.], [1., 2.]]),
              "residual": np.asarray([[-4., 4.], [4., -4.]])}
    losses = np.asarray([[.1, .8], [.7, .2]])
    value, grid = select_residual_blend(output, losses, grid=(0., 1.))
    assert value == 0
    adjusted = apply_residual_blend(output, value)
    np.testing.assert_array_equal(adjusted["score"], output["analytic_score"])


def test_anchor_candidates_preserve_top1_and_add_residual_support():
    analytic = np.asarray([[3., 2., 1.], [1., 4., 2.]])
    residual = np.asarray([[0., 1., 5.], [7., 0., 2.]])
    candidates = anchor_preserving_candidates(analytic, residual, top_k=2)
    np.testing.assert_array_equal(candidates[:, 0], analytic.argmax(1))
    np.testing.assert_array_equal(candidates[:, 1], [2, 0])
    losses = np.asarray([[.4, .3, .1], [.1, .5, .3]])
    assert candidate_recall(candidates[:, :1], losses) == 0
    assert candidate_recall(candidates, losses) == 1
