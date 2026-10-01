import numpy as np

from rcg.stable_ensemble import (fit_simplex_weights, mix_actions,
                                 select_safe_shrinkage)


def test_simplex_ensemble_is_valid_and_learns_better_action():
    labels = np.array([0, 1, 0, 1])
    weak = np.array([[.55, .45], [.45, .55], [.55, .45], [.45, .55]])
    strong = np.array([[.9, .1], [.1, .9], [.9, .1], [.1, .9]])
    actions = np.stack((weak, strong))
    weight = fit_simplex_weights(actions, labels)
    probability = mix_actions(actions, weight)
    assert (weight >= 0).all() and np.isclose(weight.sum(), 1)
    assert weight[1] > weight[0]
    assert np.allclose(probability.sum(1), 1)


def test_safe_shrinkage_can_fall_back_to_full():
    labels = np.array([0, 1])
    full = np.array([[.9, .1], [.1, .9]])
    harmful = np.array([[.1, .9], [.9, .1]])
    rho, records = select_safe_shrinkage(full, harmful, labels)
    assert rho == 0
    assert records[0]["accuracy"] == 1
