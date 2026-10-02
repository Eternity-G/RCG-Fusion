import numpy as np
import pytest
import torch

from rcg.rcg_fusion import (
    CoalitionAwareBackbone,
    CoalitionRiskPredictor,
    ConformalSafeSelector,
    build_teacher_targets,
    conformal_intervals,
    draw_coalition_masks,
    exact_shapley_from_losses,
    fit_joint_conformal,
    nonempty_coalitions,
    risk_objective,
)
from rcg.rcg_fusion_pipeline import dynamic_feature_augmentation, predict_outputs


def test_nonempty_coalitions_cover_two_and_three_modality_power_sets():
    assert nonempty_coalitions(2) == ((1, 0), (0, 1), (1, 1))
    masks = nonempty_coalitions(3)
    assert len(masks) == 7 and len(set(masks)) == 7
    assert masks[-1] == (1, 1, 1) and all(any(mask) for mask in masks)


def test_coalition_dropout_has_no_empty_input_and_half_full():
    torch.manual_seed(4)
    masks = draw_coalition_masks(40000, 3, "cpu")
    assert (masks.sum(1) > 0).all()
    assert abs(float((masks.sum(1) == 3).float().mean())-.5) < .015
    _, counts = torch.unique(masks[masks.sum(1) < 3], dim=0, return_counts=True)
    assert (counts.max()-counts.min())/counts.float().mean() < .12


def test_dynamic_feature_augmentation_is_reproducible_and_does_not_mutate_inputs():
    original = [torch.ones(128, 7), torch.ones(128, 5)]
    torch.manual_seed(19)
    first = dynamic_feature_augmentation(original, probability=1.)
    torch.manual_seed(19)
    second = dynamic_feature_augmentation(original, probability=1.)
    for source, left, right in zip(original, first, second):
        torch.testing.assert_close(source, torch.ones_like(source))
        torch.testing.assert_close(left, right)
    changed = torch.stack([(value != 1).any(1) for value in first], 1)
    assert changed.any(1).all()
    assert (changed.sum(1) == 1).all()


def test_dynamic_feature_augmentation_zero_probability_is_an_exact_copy():
    source = [torch.randn(6, 4), torch.randn(6, 3)]
    result = dynamic_feature_augmentation(source, probability=0.)
    for before, after in zip(source, result):
        torch.testing.assert_close(before, after)
        assert before.data_ptr() != after.data_ptr()


def test_backbone_rejects_empty_coalition_and_token_encoder_is_permutation_invariant():
    model = CoalitionAwareBackbone([4, 3, 2], 2, hidden=16, heads=4, dropout=0).eval()
    xs = [torch.randn(5, d) for d in (4, 3, 2)]
    with pytest.raises(ValueError):
        model(xs, torch.zeros(5, 3))
    mask = torch.tensor([[1., 0., 1.]]).repeat(5, 1)
    tokens = model.modality_tokens(xs, mask)
    permutation = torch.tensor([2, 0, 1])
    first = model.encode_tokens(tokens, mask)
    second = model.encode_tokens(tokens[:, permutation], mask[:, permutation])
    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)


def test_shapley_efficiency_and_target_direction():
    masks = nonempty_coalitions(3)
    reductions = np.array([1., -2., 3.])
    empty = np.array([10.])
    losses = np.stack([empty-np.dot(mask, reductions) for mask in masks], 1)
    phi = exact_shapley_from_losses(losses, empty, masks)
    np.testing.assert_allclose(phi[0], reductions)
    assert phi.sum() == pytest.approx(empty[0]-losses[0, -1])
    probabilities = np.full((2, 1, len(masks), 2), .5)
    teacher_losses = np.stack([losses, losses+.2])
    targets = build_teacher_targets(teacher_losses, probabilities, np.array([0]),
                                    np.array([.5, .5]), masks)
    np.testing.assert_allclose(targets["loss_mean"], losses+.1)
    assert targets["oracle_coalition_index"][0] == masks.index((1, 0, 1))
    # Adding the harmful second modality has negative edge contribution.
    relevant = [(masks[b], masks[a], targets["edge_contribution"][0, i])
                for i, (b, a) in enumerate(zip(targets["edge_base_index"], targets["edge_added_index"]))]
    assert all(value < 0 for before, after, value in relevant if before[1] == 0 and after[1] == 1)


def test_risk_predictor_is_equivariant_to_coalition_query_order():
    torch.manual_seed(2)
    model = CoalitionRiskPredictor([4, 3, 2], 3, hidden=16, heads=4, dropout=0).eval()
    xs = [torch.randn(6, d) for d in (4, 3, 2)]
    masks = torch.tensor(nonempty_coalitions(3), dtype=torch.float32)
    singleton = torch.softmax(torch.randn(6, 3, 3), -1)
    coalition = torch.softmax(torch.randn(6, 7, 3), -1)
    first = model(xs, singleton, coalition, masks)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    second = model(xs, singleton, coalition[:, permutation], masks[permutation])
    for a, b in zip(first, second):
        torch.testing.assert_close(a[:, permutation], b, atol=1e-5, rtol=1e-5)


def test_risk_objective_is_finite_for_two_modalities_and_variable_classes():
    model = CoalitionRiskPredictor([5, 7], 10, hidden=16, heads=4)
    xs = [torch.randn(8, 5), torch.randn(8, 7)]
    masks = torch.tensor(nonempty_coalitions(2), dtype=torch.float32)
    singleton = torch.softmax(torch.randn(8, 2, 10), -1)
    coalition = torch.softmax(torch.randn(8, 3, 10), -1)
    mean, sigma, safe = model(xs, singleton, coalition, masks)
    target = torch.rand_like(mean)
    base, added = torch.tensor([0, 1]), torch.tensor([2, 2])
    edge = target[:, base]-target[:, added]
    sign = torch.where(edge > .01, 2, torch.where(edge < -.01, 0, 1)).long()
    loss, parts = risk_objective(mean, sigma, safe, target, torch.zeros_like(target),
                                 torch.zeros_like(target), base, added, sign)
    assert loss.isfinite() and all(x.isfinite() for x in parts.values())


def test_joint_conformal_uses_finite_sample_higher_quantile():
    mean = np.zeros((9, 2)); sigma = np.ones((9, 2))
    observed = np.column_stack([np.arange(1, 10), np.zeros(9)])
    calibration = fit_joint_conformal(mean, sigma, observed, alpha=.2)
    # ceil((9+1)*.8)=8: the eighth order statistic.
    assert calibration.quantile == pytest.approx(8, abs=1e-5)
    lower, upper = conformal_intervals(mean[:1], sigma[:1], calibration)
    np.testing.assert_allclose([lower[0, 0], upper[0, 0]], [-8, 8], atol=1e-5)


def test_safe_selector_switches_only_on_interval_separation_and_never_selects_empty():
    calibration = fit_joint_conformal(np.zeros((10, 3)), np.ones((10, 3)),
                                      np.zeros((10, 3)), alpha=.1)
    mean = np.array([[.1, 2., 1.], [.995, 1.1, 1.]])
    sigma = np.zeros_like(mean)+1e-4
    result = ConformalSafeSelector(calibration, margin=.01).select(mean, sigma)
    assert result["selected"][0] == 0
    assert result["selected"][1] == 2
    assert np.all(result["selected"] >= 0)


def test_alpha_zero_and_large_margin_fall_back_to_full():
    mean = np.array([[0., 1., 2.], [2., 1., 0.]])
    sigma = np.ones_like(mean)
    observed = mean.copy()
    zero = fit_joint_conformal(mean, sigma, observed, alpha=0)
    assert np.all(ConformalSafeSelector(zero, .01).select(mean, sigma)["selected"] == 2)
    finite = fit_joint_conformal(mean, sigma, observed, alpha=.1)
    assert np.all(ConformalSafeSelector(finite, 1e9).select(mean, sigma)["selected"] == 2)


def test_selector_never_chooses_an_unavailable_modality_coalition():
    mean = np.array([[-40., -10., -20., 1., -30., -40., -50.]])
    sigma = np.full_like(mean, 1e-4)
    calibration = fit_joint_conformal(np.zeros((20, 7)), np.ones((20, 7)),
                                      np.zeros((20, 7)), alpha=.1)
    masks = nonempty_coalitions(3)
    available = (1, 0, 1)
    valid = np.asarray([all(not keep or available[i] for i, keep in enumerate(mask))
                        for mask in masks])
    fallback = masks.index(available)
    result = ConformalSafeSelector(calibration, .01).select(
        mean, sigma, full_index=fallback, valid_coalitions=valid)
    assert valid[result["selected"][0]]
    assert result["selected"][0] == masks.index((1, 0, 0))


def test_deployment_coalition_inference_does_not_require_labels():
    model = CoalitionAwareBackbone([3, 2], 4, hidden=16, heads=4, dropout=0).eval()
    features_only = [np.random.randn(5, 3).astype("float32"),
                     np.random.randn(5, 2).astype("float32")]
    output = predict_outputs(model, features_only, "cpu")
    assert output["probabilities"].shape == (5, 3, 4)
    assert "losses" not in output


def test_improvement_conformal_switch_implies_positive_observed_gain_on_coverage_event():
    rng = np.random.default_rng(3)
    calibration_mean = rng.normal(size=(80, 3))
    calibration_sigma = np.full((80, 3), .1)
    calibration_loss = calibration_mean + rng.normal(scale=.03, size=(80, 3))
    calibration = fit_joint_conformal(calibration_mean, calibration_sigma,
                                      calibration_loss, alpha=.1, mode="improvement")
    mean = np.array([[.2, 1.2, 1.0]])
    sigma = np.full_like(mean, .01)
    result = ConformalSafeSelector(calibration, margin=.01).select(mean, sigma)
    assert result["selected"][0] == 0
    # Construct losses inside every reported improvement interval.
    observed_improvement = (result["improvement_lower"]+result["improvement_upper"])/2
    assert observed_improvement[0, result["selected"][0]] > .01


def test_improvement_calibration_falls_back_when_missingness_changes_reference():
    mean = np.zeros((20, 3)); sigma = np.ones((20, 3)); loss = np.zeros((20, 3))
    calibration = fit_joint_conformal(mean, sigma, loss, alpha=.1, mode="improvement")
    result = ConformalSafeSelector(calibration, .01).select(mean[:2], sigma[:2], full_index=0)
    np.testing.assert_array_equal(result["selected"], [0, 0])
