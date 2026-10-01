import numpy as np
import pytest
import torch

from rcg.benefit_fusion import (
    PosteriorStructuredBenefitEstimator,
    apply_crc,
    benefit_objective,
    crc_risk_curve,
    ensemble_candidate,
    fit_crc,
    teacher_record_weights,
)
from rcg.rcg_fusion import nonempty_coalitions
from rcg.benefit_pipeline import apply_value_offset


def test_teacher_records_sum_to_one_per_original_sample():
    weights = teacher_record_weights(5, 13)
    np.testing.assert_allclose(weights.sum(0), np.ones(13))
    assert np.unique(weights).item() == pytest.approx(.2)


def test_perfect_posterior_structured_benefit_equals_realized_ce_difference():
    torch.manual_seed(2)
    masks = torch.tensor(nonempty_coalitions(2), dtype=torch.float32)
    model = PosteriorStructuredBenefitEstimator([3, 4], 3, len(masks), hidden=16,
                                                heads=4, dropout=0).eval()
    xs = [torch.randn(5, 3), torch.randn(5, 4)]
    probabilities = torch.softmax(torch.randn(5, 3, 3), -1)
    labels = torch.tensor([0, 1, 2, 0, 1])
    posterior = torch.nn.functional.one_hot(labels, 3).float()
    output = model(xs, probabilities, masks, posterior_override=posterior)
    rows = torch.arange(5)
    expected = (probabilities[rows, :, labels].log()
                - probabilities[rows, -1, labels, None].log())
    torch.testing.assert_close(output["structured_benefit"], expected.clamp(-1, 1))


def test_benefit_model_shapes_and_objective_are_finite():
    masks = torch.tensor(nonempty_coalitions(3), dtype=torch.float32)
    model = PosteriorStructuredBenefitEstimator([2, 3, 4], 2, len(masks), hidden=16, heads=4)
    xs = [torch.randn(7, d) for d in (2, 3, 4)]
    probability = torch.softmax(torch.randn(7, 7, 2), -1)
    output = model(xs, probability, masks)
    assert output["safe_probability"].shape == (7, 7)
    assert output["posterior"].shape == (7, 2)
    loss, parts = benefit_objective(output, torch.randint(0, 2, (7,)), torch.rand(7, 7))
    assert loss.isfinite() and all(value.isfinite() for value in parts.values())


def test_candidate_is_fixed_and_excludes_full_and_missing_coalitions():
    masks = nonempty_coalitions(3)
    gain = np.array([[.8, .9, .7, .6, .5, .4, .99]])
    harm = np.zeros_like(gain)
    output = {"gain": gain, "harm": harm, "safe_probability": np.full_like(gain, .9)}
    candidate = ensemble_candidate([output, output], masks, availability=(1, 0, 1))
    chosen = candidate["candidate"][0]
    assert chosen != len(masks)-1
    assert masks[chosen][1] == 0
    assert chosen == masks.index((1, 0, 0))
    shifted = apply_value_offset(candidate, .25)
    np.testing.assert_allclose(shifted["candidate_value"], candidate["candidate_value"]+.25)
    np.testing.assert_array_equal(shifted["candidate"], candidate["candidate"])


def test_crc_curve_is_monotone_and_uses_finite_sample_correction():
    score = np.array([.1, .2, .3, .4, .5])
    eligible = np.ones(5, dtype=bool)
    benefit = np.array([-1., 1., -1., 1., 1.])
    curve = crc_risk_curve(score, eligible, benefit)
    event = np.array([row["empirical_event_risk"] for row in curve])
    assert np.all(np.diff(event) <= 1e-12)
    first = curve[0]
    assert first["crc_event_bound"] == pytest.approx(5/6*first["empirical_event_risk"]+1/6)


def test_crc_threshold_controls_both_bounds_and_no_feasible_budget_falls_back():
    score = np.linspace(0, 1, 100)
    eligible = np.ones(100, dtype=bool)
    benefit = np.r_[np.full(5, -1.), np.ones(95)]
    calibration, _ = fit_crc(score, eligible, benefit, alpha_event=.05, alpha_harm=.02)
    assert calibration.event_bound <= .05 and calibration.harm_bound <= .02
    candidate = {"candidate": np.zeros(100, dtype=int), "candidate_value": np.ones(100),
                 "candidate_safe_probability": score}
    selected = apply_crc(candidate, calibration, full_index=2)
    assert np.all(selected["selected"][~selected["switch"]] == 2)

    impossible, _ = fit_crc(score[:5], eligible[:5], benefit[:5],
                            alpha_event=.01, alpha_harm=.01)
    assert np.isinf(impossible.threshold)
    assert not apply_crc(candidate, impossible, full_index=2)["switch"].any()
