from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def task_metrics(y, p):
    confidence = p.max(1)
    correct = p.argmax(1) == y
    bins = np.minimum((confidence*10).astype(int), 9)
    counts = np.bincount(bins, minlength=10)
    ece = sum(abs(confidence[bins == b].mean()-correct[bins == b].mean())*counts[b]/len(y)
              for b in range(10) if counts[b])
    return {
        "accuracy": accuracy_score(y, p.argmax(1)),
        "macro_f1": f1_score(y, p.argmax(1), average="macro", labels=[0, 1], zero_division=0),
        "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
        "brier": float(np.square(p-np.eye(2)[y]).sum(1).mean()),
        "ece": float(ece), "ece_bin_counts": ";".join(map(str, counts)),
    }


def diagnostics(frame, epsilon, score="r", high="high"):
    r, u = frame[score].to_numpy(), frame.u.to_numpy()
    h = frame[high].to_numpy(dtype=bool)
    harmful = u < -epsilon
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(spearmanr(r, u).statistic) if len(r) > 1 else np.nan
    two_classes = len(np.unique(harmful)) == 2
    return {
        "spearman": rho,
        "hcr": float(harmful[h].mean()) if h.any() else np.nan,
        "high_mean_harm": float(np.maximum(-u[h], 0).mean()) if h.any() else np.nan,
        "negative_flip_rate": float(frame.negative_flip.mean()),
        "high_negative_flip_rate": float(frame.loc[h, "negative_flip"].mean()) if h.any() else np.nan,
        "harmful_auroc": roc_auc_score(harmful, 1-r) if two_classes else np.nan,
        "harmful_auprc": average_precision_score(harmful, 1-r) if two_classes else np.nan,
        "harmful_prevalence": float(harmful.mean()),
        "high_coverage": float(h.mean()),
        "n_unique": frame.sample_id.nunique(),
        "n_high_unique": frame.loc[h, "sample_id"].nunique(),
        "n_videos": frame.video_id.nunique(),
        "n_rows_including_corruption_repeats": len(frame),
    }


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)
    finite = np.flatnonzero(np.isfinite(pvalues))
    order = finite[np.argsort(pvalues[finite])]
    if len(order):
        adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order]*(len(order)-np.arange(len(order)))))
    return adjusted


def bootstrap_ratios(numerators, denominators, weights):
    """Same cluster multiplicities for all columns: preserves paired comparisons."""
    n = weights @ np.asarray(numerators, dtype=float)
    d = weights @ np.asarray(denominators, dtype=float)
    return np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)


def centered_bootstrap_p(draws, observed):
    values = np.asarray(draws)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    # Approximate two-sided test using a null-centered paired bootstrap distribution.
    return float((1 + (np.abs(values-observed) >= abs(observed)).sum())/(len(values)+1))
