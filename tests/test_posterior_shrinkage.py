import math

import pytest
import torch

from rcg.posterior_shrinkage import (contribution_adaptive_shrinkage,
                                     posterior_risk_certificate,
                                     posterior_shrinkage,
                                     select_alpha_by_nll,
                                     worst_case_loss_increase)


def test_zero_alpha_is_full_and_probabilities_remain_normalized():
    full = torch.tensor([[.8, .2], [.3, .7]])
    posterior = torch.tensor([[.6, .4], [.1, .9]])
    torch.testing.assert_close(posterior_shrinkage(full, posterior, 0), full)
    mixture = posterior_shrinkage(full, posterior, .25)
    torch.testing.assert_close(mixture.sum(-1), torch.ones(2))


def test_pointwise_loss_increase_obeys_deterministic_bound():
    torch.manual_seed(3)
    full = torch.softmax(torch.randn(20, 4), -1)
    posterior = torch.softmax(torch.randn(20, 4), -1)
    labels = torch.randint(0, 4, (20,))
    mixture = posterior_shrinkage(full, posterior, .25)
    rows = torch.arange(20)
    increase = -mixture[rows, labels].log()+full[rows, labels].log()
    assert torch.all(increase <= worst_case_loss_increase(.25)+1e-6)
    assert worst_case_loss_increase(.25) == pytest.approx(-math.log(.75))


def test_posterior_expected_gain_exceeds_convexity_lower_bound():
    torch.manual_seed(5)
    full = torch.softmax(torch.randn(30, 3), -1)
    posterior = torch.softmax(torch.randn(30, 3), -1)
    certificate = posterior_risk_certificate(full, posterior, .25)
    assert torch.all(certificate["posterior_expected_gain"]+1e-6
                     >= certificate["convexity_lower_bound"])


def test_invalid_alpha_and_shape_are_rejected():
    with pytest.raises(ValueError):
        worst_case_loss_increase(1.)
    with pytest.raises(ValueError):
        posterior_shrinkage(torch.ones(2, 2), torch.ones(3, 2), .25)


def test_selection_alpha_uses_nll_and_ties_fall_back_to_smaller_radius():
    full = torch.tensor([[.9, .1], [.2, .8]]).numpy()
    better = torch.tensor([[.99, .01], [.01, .99]]).numpy()
    selected, scores = select_alpha_by_nll(full, better, [0, 1], (0, .25, .5, .75))
    assert selected == .75
    assert scores[.75] < scores[0.]
    selected, _ = select_alpha_by_nll(full, full, [0, 1], (0, .25, .5))
    assert selected == 0.


def test_selection_alpha_rejects_invalid_grid_and_shapes():
    full = torch.tensor([[.9, .1]]).numpy()
    with pytest.raises(ValueError):
        select_alpha_by_nll(full, full, [0], (1.,))
    with pytest.raises(ValueError):
        select_alpha_by_nll(full, full, [0, 1])


def test_contribution_adaptive_shrinkage_is_label_free_bounded_and_coherent():
    full = torch.tensor([[.8, .2], [.4, .6]], dtype=torch.float64)
    posterior = torch.tensor([[.95, .05], [.1, .9]], dtype=torch.float64)
    result = contribution_adaptive_shrinkage(full, posterior, .75, 2.)
    assert torch.all((result["alpha"] >= 0) & (result["alpha"] <= .75))
    torch.testing.assert_close(result["probability"].sum(-1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(result["alpha"],
                               .75*result["safe_probability"].square())
    assert torch.all(result["posterior_expected_gain"]+1e-8
                     >= result["convexity_lower_bound"])
    for label in (0, 1):
        increase = (-result["probability"][:, label].log()
                    + full[:, label].log())
        assert torch.all(increase <= worst_case_loss_increase(.75)+1e-8)


def test_contribution_adaptive_shrinkage_rejects_risk_seeking_exponent():
    probability = torch.tensor([[.5, .5]])
    with pytest.raises(ValueError):
        contribution_adaptive_shrinkage(probability, probability, gamma=.5)
