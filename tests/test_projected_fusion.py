import inspect

import numpy as np
import pytest
import torch

from rcg.projected_fusion import (analytic_action, build_projection,
                                  posterior_convex_projection, project_simplex,
                                  select_alpha, shrink_probability)


def test_simplex_projection_is_nonnegative_and_sums_to_one():
    value = torch.tensor([[2., -1., .5], [-3., 4., 2.]])
    projected = project_simplex(value)
    assert torch.all(projected >= 0)
    torch.testing.assert_close(projected.sum(-1), torch.ones(2))


@pytest.mark.parametrize("coalitions,top_k", [(3, 2), (7, 3)])
def test_projection_supports_two_and_three_modalities_and_is_monotone(coalitions, top_k):
    torch.manual_seed(coalitions)
    posterior = torch.softmax(torch.randn(12, 4), -1).double()
    probability = torch.softmax(torch.randn(12, coalitions, 4), -1).double()
    score = torch.randn(12, coalitions).double()
    result = posterior_convex_projection(
        posterior, probability, score, top_k=top_k, iterations=50, learning_rate=.1)
    assert result.pool_indices.shape == (12, top_k+1)
    assert torch.all(result.pool_indices[:, -1] == coalitions-1)
    assert torch.all(result.weights >= 0)
    torch.testing.assert_close(result.weights.sum(-1), torch.ones(12, dtype=torch.double))
    assert torch.all(result.objective_history[:, 1:]
                     <= result.objective_history[:, :-1]+1e-10)
    gathered = probability.gather(
        1, result.pool_indices[..., None].expand(-1, -1, 4))
    reconstruction = (result.weights[..., None]*gathered).sum(1)
    torch.testing.assert_close(result.probability, reconstruction)


def test_top_three_excludes_full_until_full_is_appended():
    posterior = torch.tensor([[.7, .3]], dtype=torch.double)
    probability = torch.tensor([[[.8, .2], [.7, .3], [.6, .4], [.55, .45],
                                 [.5, .5], [.4, .6], [.65, .35]]], dtype=torch.double)
    score = torch.tensor([[1., 3., 2., 6., 5., 4., 100.]], dtype=torch.double)
    result = posterior_convex_projection(posterior, probability, score, top_k=3)
    assert result.pool_indices[0].tolist() == [3, 4, 5, 6]


def test_zero_shrinkage_equals_full_and_action_is_coherent():
    posterior = torch.tensor([[.6, .4], [.2, .8]])
    full = torch.tensor([[.7, .3], [.4, .6]])
    projected = torch.tensor([[.5, .5], [.1, .9]])
    torch.testing.assert_close(shrink_probability(full, projected, 0), full)
    action = analytic_action(posterior, shrink_probability(full, projected, .5), full)
    torch.testing.assert_close(action["expected_benefit"], action["gain"]-action["harm"])


def test_selection_prefers_smaller_alpha_inside_nll_tolerance():
    labels = np.array([0, 1])
    full = np.array([[.9, .1], [.1, .9]])
    projected = full.copy()
    alpha, table = select_alpha(labels, full, projected)
    assert alpha == .25 and len(table) == 4


def test_build_projection_requires_no_labels_and_returns_valid_mixture():
    assert "labels" not in inspect.signature(build_projection).parameters
    rng = np.random.default_rng(4)
    probability = rng.dirichlet(np.ones(3), size=(8, 7))
    posterior = rng.dirichlet(np.ones(3), size=8)
    gain, harm = rng.random((8, 7)), rng.random((8, 7))
    result = build_projection(posterior, probability, gain, harm, top_k=3, alpha=.5)
    np.testing.assert_allclose(result["weights"].sum(1), 1.)
    assert np.all(result["weights"] >= 0)
    np.testing.assert_allclose(result["action_probability"].sum(1), 1.)
