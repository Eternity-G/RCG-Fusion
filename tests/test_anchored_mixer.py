import torch

from rcg.anchored_mixer import (AnchoredCandidateMixer,
                                anchored_mixer_objective,
                                gather_anchored_actions, shrink_to_full)
from rcg.anchored_mixer_pipeline import (analytic_action_probability,
                                         apply_adaptive_policy)


def test_actions_keep_full_and_posterior_first():
    p = torch.tensor([[[.8, .2], [.3, .7], [.6, .4]]])
    q = torch.tensor([[.4, .6]])
    actions = gather_anchored_actions(p, q, torch.tensor([[1, 0]]))
    assert torch.allclose(actions[:, 0], p[:, -1])
    assert torch.allclose(actions[:, 1], q)
    assert torch.allclose(actions[:, 2], p[:, 1])


def test_mixer_is_convex_and_objective_is_finite():
    torch.manual_seed(3)
    actions = torch.rand(8, 5, 3); actions /= actions.sum(-1, keepdim=True)
    q = torch.rand(8, 3); q /= q.sum(-1, keepdim=True)
    model = AnchoredCandidateMixer(3)
    output = model(actions, q)
    assert torch.allclose(output["weight"].sum(1), torch.ones(8))
    assert torch.allclose(output["probability"].sum(1), torch.ones(8), atol=1e-6)
    loss, parts = anchored_mixer_objective(output, actions, torch.arange(8) % 3)
    assert torch.isfinite(loss) and all(torch.isfinite(x) for x in parts.values())
    assert torch.allclose(shrink_to_full(actions[:, 0], output["probability"], 0),
                          actions[:, 0])

    ordinary = AnchoredCandidateMixer(3, anchor_full=False)
    assert not ordinary.anchor_full
    assert ordinary.full_bias.item() == 0

    sparse = AnchoredCandidateMixer(3, anchor_full=False, normalizer="sparsemax")
    sparse_output = sparse(actions, q)
    assert torch.allclose(sparse_output["weight"].sum(1), torch.ones(8), atol=1e-6)
    assert (sparse_output["weight"] >= 0).all()
    assert torch.allclose(sparse_output["probability"].sum(1), torch.ones(8), atol=1e-6)


def test_adaptive_policy_is_bounded_and_label_free():
    full = torch.tensor([[.8, .2], [.4, .6]]).numpy()
    mixture = torch.tensor([[.6, .4], [.7, .3]]).numpy()
    posterior = torch.tensor([[.7, .3], [.2, .8]]).numpy()
    safe = analytic_action_probability(full, mixture, posterior)
    probability, alpha, repeated = apply_adaptive_policy(
        full, mixture, posterior, .75, 2.)
    assert ((safe >= 0) & (safe <= 1)).all()
    assert (alpha <= .75).all()
    assert abs(probability.sum(1)-1).max() < 1e-6
    assert (safe == repeated).all()
