import numpy as np
import torch

from rcg.posterior_analytic import (
    RelationalPosteriorEstimator,
    analytic_candidate,
    analytic_contribution,
    posterior_objective,
)
from rcg.posterior_analytic_pipeline import (calibrate_members,
                                              fit_ensemble_temperature)
from rcg.rcg_fusion import nonempty_coalitions


def test_one_hot_posterior_recovers_clipped_realized_ce_improvement():
    torch.manual_seed(7)
    probability = torch.softmax(torch.randn(8, 3, 4), -1)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    posterior = torch.nn.functional.one_hot(labels, 4).float()
    output = analytic_contribution(posterior, probability)
    rows = torch.arange(len(labels))
    expected = (probability[rows, :, labels].log()
                - probability[rows, -1, labels, None].log()).clamp(-1, 1)
    torch.testing.assert_close(output["expected_benefit"], expected)
    torch.testing.assert_close(
        output["safe_probability"], (expected > .01).to(expected.dtype))


def test_analytic_decomposition_is_coherent_and_full_is_zero():
    torch.manual_seed(8)
    posterior = torch.softmax(torch.randn(6, 3), -1)
    probability = torch.softmax(torch.randn(6, 7, 3), -1)
    output = analytic_contribution(posterior, probability)
    torch.testing.assert_close(
        output["expected_benefit"], output["gain"]-output["harm"])
    assert torch.all((output["safe_probability"] >= 0)
                     & (output["safe_probability"] <= 1))
    assert torch.all(output["gain"] >= 0) and torch.all(output["harm"] >= 0)
    for key in ("expected_benefit", "gain", "harm", "safe_probability"):
        torch.testing.assert_close(output[key][:, -1], torch.zeros(6))
    torch.testing.assert_close(output["label_benefit"][:, -1], torch.zeros(6, 3))


def test_posterior_model_and_proper_objective_are_finite():
    masks = torch.tensor(nonempty_coalitions(3), dtype=torch.float32)
    model = RelationalPosteriorEstimator([2, 3, 4], 2, len(masks), hidden=16, heads=4)
    output = model([torch.randn(9, d) for d in (2, 3, 4)],
                   torch.softmax(torch.randn(9, 7, 2), -1), masks)
    loss, parts = posterior_objective(output["posterior_logits"],
                                      torch.randint(0, 2, (9,)))
    assert output["posterior"].shape == (9, 2)
    assert loss.isfinite() and all(value.isfinite() for value in parts.values())


def test_candidate_excludes_full_and_unavailable_modalities_and_is_deterministic():
    masks = nonempty_coalitions(3)
    probability = np.array([[
        [.8, .2], [.7, .3], [.6, .4], [.75, .25],
        [.65, .35], [.55, .45], [.7, .3],
    ]])
    posterior = [np.array([[.9, .1]]), np.array([[.8, .2]])]
    first = analytic_candidate(posterior, probability, masks, availability=(1, 0, 1))
    second = analytic_candidate(posterior, probability, masks, availability=(1, 0, 1))
    chosen = first["candidate"][0]
    assert chosen != len(masks)-1 and masks[chosen][1] == 0
    np.testing.assert_array_equal(first["candidate"], second["candidate"])
    assert np.isfinite(first["candidate_expected_std"]).all()


def test_ensemble_temperature_is_deterministic_and_preserves_probabilities():
    posteriors = [np.array([[.8, .2], [.3, .7], [.6, .4]]),
                  np.array([[.7, .3], [.4, .6], [.55, .45]])]
    labels = np.array([0, 1, 1])
    first = fit_ensemble_temperature(posteriors, labels)
    second = fit_ensemble_temperature(posteriors, labels)
    assert first == second and first > 0
    calibrated = calibrate_members(posteriors, first)
    np.testing.assert_allclose(np.stack(calibrated).sum(-1), 1.)


def test_inference_signature_has_no_label_argument():
    assert "labels" not in RelationalPosteriorEstimator.forward.__annotations__
