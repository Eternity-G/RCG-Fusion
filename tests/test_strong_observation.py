import numpy as np
import torch

from rcg.strong_models import build_model
from rcg.strong_observation import all_masks, exact_shapley


def test_masks_have_all_nonempty_coalitions():
    masks=all_masks(3)
    assert len(masks)==7
    assert (0,0,0) not in masks and (1,1,1) in masks


def test_exact_shapley_efficiency_for_additive_losses():
    # Value=-loss; adding modalities gives values 1, 2, and -0.5.
    effects=np.array([1.,2.,-.5]); losses={}
    for mask in all_masks(3,include_empty=True):
        losses[mask]=np.array([5.-np.dot(mask,effects)])
    phi=exact_shapley(losses,3)
    np.testing.assert_allclose(phi[0],effects)
    np.testing.assert_allclose(phi.sum(1),losses[(0,0,0)]-losses[(1,1,1)])


def test_all_strong_models_support_missing_coalitions_and_backward():
    xs=[torch.randn(6,7),torch.randn(6,5),torch.randn(6,3)]
    mask=torch.tensor([[1,1,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1]],dtype=torch.float32)
    y=torch.tensor([0,1,0,1,0,1])
    for name in ("concat","tmc","qmf","pdf","i2moe"):
        model=build_model(name,[7,5,3],2,hidden=16)
        out=model(xs,mask)
        assert out["logits"].shape==(6,2)
        assert out["native_score"].shape==(6,3)
        loss=torch.nn.functional.cross_entropy(out["logits"],y)
        if hasattr(model,"auxiliary"): loss=loss+model.auxiliary(out,y,mask)
        loss.backward()
