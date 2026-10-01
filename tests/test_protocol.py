import numpy as np
import pandas as pd
import pytest
import torch

from rcg.data import normalize, pool_features, video_id
from rcg.evaluate import Condition, corrupt, conditions
from rcg.models import probability_fusion, ds_pair, ds_fusion, draw_presence, Concat, evidential_loss
from rcg.stats import diagnostics, holm, bootstrap_ratios, task_metrics
from rcg.train import fit_temperature
from rcg.coalition import coalition_masks, exact_shapley, coalition_statistics
from rcg.contribution import RelationalLossPredictor, estimator_objective, sparsemax, CANDIDATES


def test_high_confidence_correct_can_have_negative_contribution():
    # Both predictors correct; the .9 predictor dilutes the stronger .99 one.
    p = torch.tensor([[[.1, .9], [.01, .99]]], dtype=torch.float64)
    full = probability_fusion(p, torch.ones(1, 2))
    without = probability_fusion(p, torch.tensor([[0., 1.]]))
    u = torch.log(full[0, 1])-torch.log(without[0, 1])
    assert u.item() == pytest.approx(np.log(.945/.99))
    assert u < -.01


def test_weighted_deletion_renormalizes_and_ignores_removed_input():
    p = torch.tensor([[[.9, .1], [.2, .8], [.4, .6]]])
    mask = torch.tensor([[0., 1., 1.]])
    out = probability_fusion(p, mask, True)
    expected = (.8*p[:, 1]+.6*p[:, 2])/1.4
    torch.testing.assert_close(out, expected)
    p[:, 0] = torch.tensor([0., 1.])
    torch.testing.assert_close(probability_fusion(p, mask, True), expected)
    with pytest.raises(ValueError):
        probability_fusion(p, torch.zeros_like(mask))


def test_context_corruption_preserves_target_and_is_severity_nested():
    x = [np.ones((8, 3), np.float32) for _ in range(3)]
    a, _ = corrupt(x, Condition("context", "gaussian", "audio", .5, 101))
    b, _ = corrupt(x, Condition("context", "gaussian", "audio", 1., 101))
    np.testing.assert_array_equal(a[1], x[1])
    np.testing.assert_allclose(b[0]-x[0], 2*(a[0]-x[0]), atol=2e-7)
    assert not np.array_equal(a[2], x[2])


def test_zero_noise_and_zero_mask_identity():
    x = [np.arange(15, dtype=np.float32).reshape(5, 3) for _ in range(3)]
    for kind in ("gaussian", "mask"):
        out, _ = corrupt(x, Condition("main", kind, "text", 0., 101))
        for a, b in zip(x, out):
            np.testing.assert_array_equal(a, b)


def test_pooling_uses_metadata_not_padded_values():
    x = np.array([[[90., 90.], [1., 3.], [3., 5.], [90., 90.], [100., 100.]]], dtype=np.float32)
    bert = np.zeros((1, 3, 5), dtype=int)
    bert[0, 1] = [1, 1, 1, 1, 0]
    bert[0, 0, [0, 3]] = [101, 102]
    out, _ = pool_features({"text": x, "text_bert": bert}, "text")
    np.testing.assert_allclose(out, [[2, 4]])


def test_nonfinite_frames_do_not_poison_pooling():
    x = np.array([[[2., 4.], [np.inf, 0.]]], dtype=np.float32)
    out, audit = pool_features({"audio": x, "audio_lengths": [2]}, "audio")
    np.testing.assert_allclose(out, [[2, 4]])
    assert audit["nonfinite_frames_excluded"] == 1


def test_normalization_train_only():
    train = np.array([[0., 2.], [2., 2.]], dtype=np.float32)
    values, mean, std = normalize(train, [train, np.array([[100., 2.]], np.float32)])
    np.testing.assert_allclose(mean, [1, 2])
    np.testing.assert_allclose(values[1], [[99, 0]])


def test_source_video_parser():
    assert video_id("abc$_$17") == "abc"
    assert video_id(["abc", "17"]) == "abc"
    with pytest.raises(ValueError):
        video_id("ambiguous")


def test_missingness_schedule_has_no_empty_combination():
    torch.manual_seed(5)
    masks = draw_presence(40000, "cpu")
    assert torch.all(masks.sum(1) > 0)
    assert abs(float((masks.sum(1) == 3).float().mean())-.5) < .015
    unique, counts = torch.unique(masks[masks.sum(1) != 3], dim=0, return_counts=True)
    assert len(unique) == 6
    assert (counts.max()-counts.min())/counts.float().mean() < .15


def test_masked_concat_ignores_deleted_values():
    model = Concat([2, 3, 4]).eval()
    xs = [torch.randn(5, d) for d in [2, 3, 4]]
    mask = torch.ones(5, 3)
    mask[:, 1] = 0
    a = model(xs, mask)
    xs[1] += 10000
    torch.testing.assert_close(a, model(xs, mask))


def test_tmc_vacuous_identity_and_deletion():
    a = torch.tensor([[3., 4.], [2., 7.]])
    b = torch.tensor([[2., 9.], [4., 2.]])
    torch.testing.assert_close(ds_pair(a, torch.ones_like(a)), a)
    alphas = torch.stack([a, b], 1)
    torch.testing.assert_close(ds_fusion(alphas, torch.tensor([[1., 0.], [1., 0.]])), a)
    torch.testing.assert_close(ds_pair(a, b), ds_pair(b, a))
    assert evidential_loss(a, torch.tensor([0, 1]), 1).isfinite()


def test_tmc_matches_independent_closed_form():
    # For Dirichlet evidence under DS: e_combined=e1+e2+e1*e2/K.
    a = torch.tensor([[2., 5.]], dtype=torch.float64)
    b = torch.tensor([[4., 3.]], dtype=torch.float64)
    expected = 1+(a-1)+(b-1)+(a-1)*(b-1)/2
    torch.testing.assert_close(ds_pair(a, b), expected)


def test_calibration_uses_nll_and_improves_calibration_objective():
    logits = np.array([[0., 8.], [0., 8.], [0., 8.], [0., 8.]])
    t, info = fit_temperature(logits, np.array([1, 1, 1, 0]))
    assert t > 1
    assert info["after_nll"] < info["before_nll"]


def test_harm_metrics_and_one_class_auroc():
    f = pd.DataFrame({"r": [.99, .98, .7], "u": [-.1, .001, -.5], "high": [True, True, False],
                      "sample_id": ["a", "b", "c"], "video_id": ["v", "v", "w"], "negative_flip": [True, False, False]})
    result = diagnostics(f, .01)
    assert result["hcr"] == .5
    assert result["n_high_unique"] == 2
    assert result["high_mean_harm"] == .05
    f.u = -.2
    assert np.isnan(diagnostics(f, .01)["harmful_auroc"])


def test_cluster_bootstrap_does_not_treat_repeats_as_new_clusters():
    weights = np.array([[2, 0], [1, 1], [0, 2]], dtype=float)
    n = np.array([[1.], [0.]])
    d = np.array([[2.], [1.]])
    np.testing.assert_allclose(bootstrap_ratios(n, d, weights), bootstrap_ratios(3*n, 3*d, weights))
    np.testing.assert_allclose(bootstrap_ratios(n, d, weights).ravel(), [.5, 1/3, 0])


def test_holm_and_ece_boundary():
    np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
    result = task_metrics(np.array([0, 1]), np.eye(2))
    assert result["ece"] == 0
    assert result["accuracy"] == 1


def test_registered_condition_count():
    from rcg.io import load_json
    cfg = load_json("config.json")
    c = list(conditions(cfg))
    assert len(c) == 103
    assert len({(x.key, x.corruption_seed) for x in c}) == 103


def test_coalition_masks_cover_power_set_once():
    masks = coalition_masks()
    assert len(masks) == 8
    assert len(set(masks)) == 8
    assert masks[0] == (0, 0, 0)
    assert masks[-1] == (1, 1, 1)


def test_exact_shapley_efficiency_and_signed_contribution():
    # Additive loss reductions of +1, -2 and +3. The second modality is harmful.
    reductions = np.array([1., -2., 3.])
    losses = {}
    for mask in coalition_masks():
        losses[mask] = np.array([10 - np.dot(mask, reductions)])
    phi = exact_shapley(losses, (1, 1, 1))
    np.testing.assert_allclose(phi[0], reductions)
    assert phi.sum() == pytest.approx(losses[(0, 0, 0)][0] - losses[(1, 1, 1)][0])
    shapley, deletion, regret, oracle = coalition_statistics(losses, (1, 1, 1))
    np.testing.assert_allclose(shapley, deletion)
    assert regret[0] == pytest.approx(2.)
    assert oracle[0] == "TV"


def test_shapley_handles_missing_modality():
    available = (1, 0, 1)
    losses = {
        (0, 0, 0): np.array([3.]),
        (1, 0, 0): np.array([2.]),
        (0, 0, 1): np.array([2.5]),
        (1, 0, 1): np.array([1.]),
    }
    phi, deletion, regret, oracle = coalition_statistics(losses, available)
    assert np.isnan(phi[0, 1]) and np.isnan(deletion[0, 1])
    assert phi[0, 0] + phi[0, 2] == pytest.approx(2.)
    assert regret[0] == 0
    assert oracle[0] == "TV"


def test_relational_predictor_shapes_and_finite_objective():
    model = RelationalLossPredictor([4, 3, 2], hidden=8)
    xs = [torch.randn(6, d) for d in (4, 3, 2)]
    probs = torch.softmax(torch.randn(6, 3, 2), -1)
    coalition_probs = torch.softmax(torch.randn(6, 7, 2), -1)
    masks = torch.tensor(CANDIDATES, dtype=torch.float32)
    mean, logvar = model(xs, probs, masks, coalition_probs)
    assert mean.shape == logvar.shape == (6, 7)
    assert (mean >= 0).all()
    loss, parts = estimator_objective(mean, logvar, torch.rand(6, 7), masks)
    assert loss.isfinite() and all(value.isfinite() for value in parts.values())


def test_sparsemax_is_sparse_probability_map():
    output = sparsemax(torch.tensor([[3., 1., -2.], [1., 1., 1.]]), 1)
    torch.testing.assert_close(output.sum(1), torch.ones(2))
    assert output[0, 1] == 0 and output[0, 2] == 0
    torch.testing.assert_close(output[1], torch.full((3,), 1/3))
