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


def test_unavailable_modalities_and_invalid_coalitions_do_not_change_posterior():
    torch.manual_seed(41)
    masks = torch.tensor(nonempty_coalitions(3), dtype=torch.float32)
    model = RelationalPosteriorEstimator(
        [4, 3, 2], classes=2, n_coalitions=len(masks), hidden=16,
        heads=4, layers=1, dropout=0).eval()
    xs = [torch.randn(5, 4), torch.randn(5, 3), torch.randn(5, 2)]
    probability = torch.softmax(torch.randn(5, len(masks), 2), -1)
    availability = torch.tensor([[1., 0., 1.]]).repeat(5, 1)
    first = model(xs, probability, masks, availability)["posterior"]
    changed_x = [xs[0], torch.randn_like(xs[1])*100, xs[2]]
    changed_probability = probability.clone()
    invalid = ~((masks <= availability[0]).all(1))
    changed_probability[:, invalid] = torch.softmax(
        torch.randn_like(changed_probability[:, invalid])*100, -1)
    second = model(changed_x, changed_probability, masks, availability)["posterior"]
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)


def test_all_available_path_matches_legacy_posterior_path():
    torch.manual_seed(43)
    masks = torch.tensor(nonempty_coalitions(2), dtype=torch.float32)
    model = RelationalPosteriorEstimator(
        [3, 4], classes=3, n_coalitions=len(masks), hidden=16,
        heads=4, layers=1, dropout=0).eval()
    xs = [torch.randn(7, 3), torch.randn(7, 4)]
    probability = torch.softmax(torch.randn(7, len(masks), 3), -1)
    legacy = model(xs, probability, masks)["posterior"]
    explicit = model(xs, probability, masks, torch.ones(7, 2))["posterior"]
    torch.testing.assert_close(legacy, explicit, atol=1e-6, rtol=1e-6)


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
