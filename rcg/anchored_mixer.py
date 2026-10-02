"""Full-coalition anchored mixture for contribution-ranked candidates."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def gather_anchored_actions(probabilities: torch.Tensor, posterior: torch.Tensor,
                            candidates: torch.Tensor) -> torch.Tensor:
    """Return [full, posterior, analytic anchor, residual support ...] actions."""
    if probabilities.ndim != 3 or posterior.ndim != 2 or candidates.ndim != 2:
        raise ValueError("expected probabilities [B,C,K], posterior [B,K], candidates [B,R]")
    if len(probabilities) != len(posterior) or len(probabilities) != len(candidates):
        raise ValueError("batch dimensions differ")
    rows = torch.arange(len(probabilities), device=probabilities.device)[:, None]
    selected = probabilities[rows, candidates]
    return torch.cat((probabilities[:, -1:, :], posterior[:, None, :], selected), dim=1)


class AnchoredCandidateMixer(nn.Module):
    """Score actions relative to a full-coalition anchor.

    The full prediction is always action zero.  The learned result remains a
    convex combination of valid predictions, and can subsequently be shrunk
    toward action zero on the selection split.
    """

    def __init__(self, classes: int, hidden: int = 64, *, anchor_full: bool = True):
        super().__init__()
        # p, log(p), q, log(q), |p-q|, H(p), H(q), action type/rank (5)
        width = classes * 5 + 2 + 5
        self.scorer = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                    nn.Dropout(.15), nn.Linear(hidden, 1))
        self.anchor_full = bool(anchor_full)
        if self.anchor_full:
            self.full_bias = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("full_bias", torch.tensor(0.0))

    def forward(self, actions: torch.Tensor, posterior: torch.Tensor):
        if actions.ndim != 3 or posterior.ndim != 2:
            raise ValueError("expected actions [B,A,K] and posterior [B,K]")
        b, a, _ = actions.shape
        q = posterior[:, None].expand(-1, a, -1)
        p = actions.clamp_min(1e-6)
        q_safe = q.clamp_min(1e-6)
        entropy_p = -(p * p.log()).sum(-1, keepdim=True)
        entropy_q = -(q_safe * q_safe.log()).sum(-1, keepdim=True)
        kind = torch.zeros((b, a, 5), dtype=p.dtype, device=p.device)
        kind[:, 0, 0] = 1                         # full anchor
        if a > 1:
            kind[:, 1, 1] = 1                     # posterior action
        for index in range(2, a):
            kind[:, index, min(index, 4)] = 1      # ranked coalition action
        feature = torch.cat((p, p.log(), q, q_safe.log(), (p-q).abs(),
                             entropy_p, entropy_q, kind), dim=-1)
        logits = self.scorer(feature).squeeze(-1)
        logits[:, 0] = logits[:, 0] + self.full_bias
        weight = logits.softmax(-1)
        mixture = (weight[:, :, None] * actions).sum(1)
        return {"probability": mixture, "weight": weight, "logits": logits}


def anchored_mixer_objective(output: dict[str, torch.Tensor], actions: torch.Tensor,
                             labels: torch.Tensor, *, oracle_temperature: float = .1,
                             harm_weight: float = .5, oracle_weight: float = .25):
    """Optimize final NLL plus asymmetric harm and soft-oracle distillation."""
    rows = torch.arange(len(labels), device=labels.device)
    probability = output["probability"].clamp_min(1e-8)
    label_index = labels[:, None, None].expand(-1, actions.shape[1], 1)
    action_loss = -actions.clamp_min(1e-8).log().gather(2, label_index).squeeze(-1)
    final_loss = -probability.log()[rows, labels]
    full_loss = action_loss[:, 0]
    harm = F.relu(final_loss-full_loss)
    soft_oracle = F.softmax(-action_loss/oracle_temperature, dim=1)
    distillation = -(soft_oracle*output["weight"].clamp_min(1e-8).log()).sum(1)
    loss = final_loss.mean()+harm_weight*harm.square().mean()+oracle_weight*distillation.mean()
    return loss, {"nll": final_loss.mean(), "harm": harm.mean(),
                  "oracle_distillation": distillation.mean()}


def shrink_to_full(full: torch.Tensor, mixture: torch.Tensor, rho: float) -> torch.Tensor:
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    return (1-float(rho))*full+float(rho)*mixture
