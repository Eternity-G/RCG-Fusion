"""Unified-feature adaptations of strong dynamic-fusion baselines.

These models keep the published scoring mechanism while replacing dataset-
specific encoders with small heads over the same frozen features.  Every model
accepts an explicit coalition mask, which makes all counterfactual coalitions
valid training inputs.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.masked_fill(~mask.bool(), -1e9)
    return torch.softmax(values, dim=1)


class Projectors(nn.Module):
    def __init__(self, dims, hidden=128, dropout=.2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)) for d in dims
        ])

    def forward(self, xs, mask):
        return [layer(x) * mask[:, i:i+1] for i, (layer, x) in enumerate(zip(self.layers, xs))]


class CoalitionConcat(nn.Module):
    method = "coalition_concat"

    def __init__(self, dims, classes, hidden=128, dropout=.2):
        super().__init__()
        self.proj = Projectors(dims, hidden, dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden*len(dims)+len(dims), hidden), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, classes))
        self.modality_heads = nn.ModuleList([nn.Linear(hidden, classes) for _ in dims])

    def forward(self, xs, mask):
        features = self.proj(xs, mask)
        mono = torch.stack([head(z) for head, z in zip(self.modality_heads, features)], 1)
        logits = self.classifier(torch.cat(features + [mask], 1))
        score = mono.softmax(-1).max(-1).values
        return {"logits": logits, "mono_logits": mono, "native_score": score,
                "native_score_name": "maximum_probability"}

    @staticmethod
    def auxiliary(output, y, mask):
        mono=output["mono_logits"]
        losses=F.cross_entropy(mono.flatten(0,1),y[:,None].expand(-1,mono.shape[1]).reshape(-1),
                               reduction="none").reshape(mask.shape)
        return (losses*mask).sum()/mask.sum().clamp_min(1)


class QMF(nn.Module):
    """QMF unified-feature adaptation using energy confidence and rank loss."""
    method = "qmf"

    def __init__(self, dims, classes, hidden=128, dropout=.2):
        super().__init__()
        self.proj = Projectors(dims, hidden, dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, classes) for _ in dims])

    def forward(self, xs, mask):
        features = self.proj(xs, mask)
        mono = torch.stack([head(z) for head, z in zip(self.heads, features)], 1)
        # Exact score used by the official QMF late-fusion implementation.
        quality = torch.logsumexp(mono, dim=-1) / 10.0
        weights = quality * mask
        logits = (mono * weights.detach().unsqueeze(-1)).sum(1)
        return {"logits": logits, "mono_logits": mono, "native_score": quality,
                "native_score_name": "qmf_energy_confidence"}

    @staticmethod
    def auxiliary(output, y, mask):
        mono, score = output["mono_logits"], output["native_score"]
        losses = F.cross_entropy(mono.flatten(0, 1), y[:, None].expand(-1, mono.shape[1]).reshape(-1),
                                 reduction="none").reshape_as(score)
        total = (losses * mask).sum() / mask.sum().clamp_min(1)
        # Within-batch pairwise ranking: lower task loss should receive higher confidence.
        rolled_loss, rolled_score = losses.roll(1, 0), score.roll(1, 0)
        target = torch.sign(rolled_loss - losses)
        valid = (mask * mask.roll(1, 0)).bool() & target.ne(0)
        if valid.any():
            total = total + F.margin_ranking_loss(score[valid], rolled_score[valid], target[valid], margin=0.)
        return total


class PDF(nn.Module):
    """PDF Mono-/Holo-Confidence and calibrated Co-Belief adaptation."""
    method = "pdf"

    def __init__(self, dims, classes, hidden=128, dropout=.2):
        super().__init__()
        self.proj = Projectors(dims, hidden, dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, classes) for _ in dims])
        self.confid = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden*2), nn.ReLU(), nn.Linear(hidden*2, hidden),
                          nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid()) for _ in dims
        ])

    def forward(self, xs, mask):
        features = self.proj(xs, mask)
        mono_logits = torch.stack([head(z) for head, z in zip(self.heads, features)], 1)
        mono_conf = torch.cat([net(z) for net, z in zip(self.confid, features)], 1).clamp(1e-5, 1-1e-5)
        log_conf = mono_conf.log() * mask
        total_log = log_conf.sum(1, keepdim=True)
        # This reduces exactly to the paper's two-modality holo confidence.
        holo = (total_log-log_conf) / total_log.clamp_max(-1e-6)
        cobelief = (mono_conf + holo) * mask
        prob = mono_logits.softmax(-1)
        dynamic_uncertainty = (prob - 1/prob.shape[-1]).abs().mean(-1)
        max_du = dynamic_uncertainty.masked_fill(~mask.bool(), -1).max(1, keepdim=True).values.clamp_min(1e-8)
        relative = (dynamic_uncertainty/max_du).clamp(max=1) * mask
        calibrated_cobelief = cobelief * relative
        weights = masked_softmax(calibrated_cobelief, mask)
        logits = (mono_logits * weights.detach().unsqueeze(-1)).sum(1)
        return {"logits": logits, "mono_logits": mono_logits, "native_score": calibrated_cobelief,
                "mono_confidence": mono_conf, "holo_confidence": holo,
                "cobelief": cobelief, "fusion_weight": weights,
                "native_score_name": "pdf_calibrated_cobelief"}

    @staticmethod
    def auxiliary(output, y, mask):
        mono = output["mono_logits"]
        ce = F.cross_entropy(mono.flatten(0, 1), y[:, None].expand(-1, mono.shape[1]).reshape(-1),
                             reduction="none").reshape(mask.shape)
        target = mono.softmax(-1).gather(2, y[:, None, None].expand(-1, mono.shape[1], 1)).squeeze(-1).detach()
        confidence_loss = (torch.abs(output["mono_confidence"]-target)*mask).sum()/mask.sum().clamp_min(1)
        return (ce*mask).sum()/mask.sum().clamp_min(1) + confidence_loss


def ds_pair(a, b):
    k = a.shape[-1]
    sa, sb = a.sum(-1, keepdim=True), b.sum(-1, keepdim=True)
    ba, bb, ua, ub = (a-1)/sa, (b-1)/sb, k/sa, k/sb
    conflict = ba.sum(-1, keepdim=True)*bb.sum(-1, keepdim=True)-(ba*bb).sum(-1, keepdim=True)
    denominator = (1-conflict).clamp_min(1e-10)
    belief = (ba*bb + ba*ub + bb*ua)/denominator
    uncertainty = ua*ub/denominator
    return belief*(k/uncertainty.clamp_min(1e-10))+1


class TMCGeneric(nn.Module):
    method = "tmc"

    def __init__(self, dims, classes, hidden=128, dropout=.2):
        super().__init__()
        self.classes = classes
        self.proj = Projectors(dims, hidden, dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, classes) for _ in dims])

    def forward(self, xs, mask):
        features = self.proj(xs, mask)
        evidence = torch.stack([F.softplus(head(z))+1 for head, z in zip(self.heads, features)], 1)
        fused = torch.ones_like(evidence[:, 0])
        for i in range(evidence.shape[1]):
            alpha = torch.where(mask[:, i:i+1].bool(), evidence[:, i], torch.ones_like(fused))
            fused = ds_pair(fused, alpha)
        score = (1-self.classes/evidence.sum(-1))*mask
        return {"logits": fused.log(), "alpha": evidence, "native_score": score,
                "native_score_name": "tmc_evidence_reliability"}

    def auxiliary(self, output, y, mask):
        alpha = output["alpha"]
        probs = alpha/alpha.sum(-1, keepdim=True)
        loss = F.nll_loss(output["logits"].log_softmax(-1), y)
        mono = -probs.gather(2, y[:, None, None].expand(-1, alpha.shape[1], 1)).clamp_min(1e-12).log().squeeze(-1)
        return loss + (mono*mask).sum()/mask.sum().clamp_min(1)


class I2MoE(nn.Module):
    """I²MoE interaction-expert adaptation over fixed-dimensional features."""
    method = "i2moe"

    def __init__(self, dims, classes, hidden=128, dropout=.2):
        super().__init__()
        self.n = len(dims)
        self.proj = Projectors(dims, hidden, dropout)
        input_dim = hidden*self.n+self.n
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, classes))
            for _ in range(self.n+2)
        ])
        self.router = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(hidden, self.n+2))

    def _encoded(self, xs, mask):
        features = self.proj(xs, mask)
        return torch.cat(features+[mask], 1)

    def forward(self, xs, mask):
        encoded = self._encoded(xs, mask)
        expert_logits = torch.stack([expert(encoded) for expert in self.experts], 1)
        route = torch.softmax(self.router(encoded), 1)
        logits = (expert_logits*route.unsqueeze(-1)).sum(1)
        # First n routes are the paper's modality-uniqueness experts.
        return {"logits": logits, "expert_logits": expert_logits, "route": route,
                "native_score": route[:, :self.n]*mask,
                "native_score_name": "i2moe_uniqueness_route"}

    def auxiliary(self, output, y, mask):
        experts = output["expert_logits"]
        labels = y[:, None].expand(-1, experts.shape[1]).reshape(-1)
        return F.cross_entropy(experts.flatten(0, 1), labels)

    def interaction_regularizer(self, xs, mask):
        """Official uniqueness/synergy/redundancy objectives under replacement."""
        base = self.forward(xs, mask)["expert_logits"]
        replacements = []
        for m in range(self.n):
            changed = list(xs)
            changed[m] = torch.randn_like(changed[m])
            replacements.append(self.forward(changed, mask)["expert_logits"])
        terms = []
        for m in range(self.n):
            anchor = base[:, m]
            negative = replacements[m][:, m]
            positives = [replacements[j][:, m] for j in range(self.n) if j != m]
            if positives:
                terms.append(torch.stack([F.triplet_margin_loss(anchor, pos, negative) for pos in positives]).mean())
        synergy_anchor = F.normalize(base[:, self.n], dim=1)
        terms.append(torch.stack([
            (synergy_anchor*F.normalize(item[:, self.n], dim=1)).sum(1).mean() for item in replacements
        ]).mean())
        redundancy_anchor = F.normalize(base[:, self.n+1], dim=1)
        terms.append(torch.stack([
            (1-(redundancy_anchor*F.normalize(item[:, self.n+1], dim=1)).sum(1)).mean()
            for item in replacements
        ]).mean())
        return torch.stack(terms).sum()


MODELS = {"concat": CoalitionConcat, "qmf": QMF, "pdf": PDF, "tmc": TMCGeneric, "i2moe": I2MoE}


def build_model(name, dims, classes, hidden=128, dropout=.2):
    return MODELS[name](dims, classes, hidden, dropout)
