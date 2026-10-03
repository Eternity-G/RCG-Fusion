import numpy as np

from rcg.final_system import (METHOD_VERSION, fit_final_system,
                              method_manifest, prediction_hash)


def probabilities(seed, members=3, samples=11, classes=4):
    rng = np.random.default_rng(seed)
    value = rng.dirichlet(np.ones(classes), size=(members, samples))
    return value


def test_final_system_is_convex_reproducible_and_label_free_at_test():
    selection_full = probabilities(1)
    selection_a7 = probabilities(2)
    test_full = probabilities(3, samples=7)
    test_a7 = probabilities(4, samples=7)
    labels = np.arange(11) % 4
    first = fit_final_system(selection_full, selection_a7, labels,
                             test_full, test_a7, member_ids=(11, 22, 33))
    second = fit_final_system(selection_full, selection_a7, labels,
                              test_full, test_a7, member_ids=(11, 22, 33))
    np.testing.assert_allclose(first.final_probability, second.final_probability)
    np.testing.assert_allclose(first.action_weights.sum(), 1.)
    assert np.all(first.action_weights >= 0)
    assert 0 <= first.fallback_rho <= 1
    np.testing.assert_allclose(first.final_probability.sum(1), 1.)
    assert first.action_names[0] == "member_11_full"
    assert first.action_names[1] == "member_11_a7"


def test_zero_fallback_recovers_full_ensemble():
    full = probabilities(5, members=2, samples=8, classes=3)
    a7 = probabilities(6, members=2, samples=8, classes=3)
    labels = full.mean(0).argmax(1)
    result = fit_final_system(full, a7, labels, full, a7,
                              fallback_grid=(0.,), member_ids=(1, 2))
    assert result.fallback_rho == 0
    np.testing.assert_allclose(result.final_probability, full.mean(0))


def test_prediction_hash_is_order_invariant_but_probability_sensitive():
    ids = np.array(["b", "a", "c"])
    folds = np.array([1, 0, 1])
    probability = np.array([[.8, .2], [.3, .7], [.5, .5]])
    first = prediction_hash(ids, folds, probability)
    order = np.array([2, 0, 1])
    second = prediction_hash(ids[order], folds[order], probability[order])
    assert first == second
    changed = probability.copy(); changed[0, 0] += 1e-8; changed[0, 1] -= 1e-8
    assert prediction_hash(ids, folds, changed) != first
    assert method_manifest()["method_version"] == METHOD_VERSION
