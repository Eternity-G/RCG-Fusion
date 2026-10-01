from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


class Concat(nn.Module):
    def __init__(self, dims, hidden=128, dropout=0.2):
        super().__init__()
        self.head = Head(sum(dims) + len(dims), hidden, dropout)

    def forward(self, xs, present):
        return self.head(torch.cat([x * present[:, i:i+1] for i, x in enumerate(xs)] + [present], 1))


def draw_presence(n, device):
    combinations = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                 [1, 1, 0], [1, 0, 1], [0, 1, 1]], device=device, dtype=torch.float32)
    mask = combinations[torch.randint(6, (n,), device=device)].clone()
    mask[torch.rand(n, device=device) < 0.5] = 1
    return mask


def probability_fusion(probabilities, present, weighted=False):
    weights = probabilities.max(-1).values if weighted else torch.ones_like(present)
    weights = weights * present
    if torch.any(weights.sum(1) <= 0):
        raise ValueError("At least one modality must be present")
    weights = weights / weights.sum(1, keepdim=True)
    return (probabilities * weights[:, :, None]).sum(1)


def ds_pair(a, b):
    """Dempster-Shafer opinion combination, K=2. Algebra from the TMC paper."""
    k = a.shape[-1]
    sa, sb = a.sum(-1, keepdim=True), b.sum(-1, keepdim=True)
    ba, bb = (a-1)/sa, (b-1)/sb
    ua, ub = k/sa, k/sb
    # Sum off-diagonal products; clamp only the numerical denominator.
    conflict = ba.sum(-1, keepdim=True)*bb.sum(-1, keepdim=True) - (ba*bb).sum(-1, keepdim=True)
    denominator = (1-conflict).clamp_min(1e-10)
    belief = (ba*bb + ba*ub + bb*ua) / denominator
    uncertainty = ua*ub / denominator
    return belief * (k/uncertainty.clamp_min(1e-10)) + 1


def ds_fusion(alphas, present):
    # Uniform Dirichlet is a vacuous opinion and the identity for DS combination.
    output = torch.ones_like(alphas[:, 0])
    for m in range(alphas.shape[1]):
        a = torch.where(present[:, m:m+1].bool(), alphas[:, m], torch.ones_like(output))
        output = ds_pair(output, a)
    return output


def evidential_loss(alpha, y, annealing):
    target = F.one_hot(y, alpha.shape[1]).float()
    fit = (target * (torch.digamma(alpha.sum(1, keepdim=True))-torch.digamma(alpha))).sum(1)
    regularized = (alpha-1)*(1-target)+1
    s = regularized.sum(1, keepdim=True)
    k = alpha.shape[1]
    kl = (torch.lgamma(s).squeeze(1)-torch.lgamma(regularized).sum(1)
          - torch.lgamma(torch.tensor(float(k), device=alpha.device))
          + ((regularized-1)*(torch.digamma(regularized)-torch.digamma(s))).sum(1))
    return (fit + annealing*kl).mean()


class TMC(nn.Module):
    def __init__(self, dims, hidden=128, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(d, hidden, dropout) for d in dims])

    def forward(self, xs):
        # Softplus is a nonnegative evidence activation; explicitly documented adaptation.
        return torch.stack([F.softplus(head(x))+1 for head, x in zip(self.heads, xs)], 1)
