"""Paired bootstrap (corpus BLEU) and exact McNemar test."""
from __future__ import annotations
import numpy as np
from scipy.stats import binomtest


def _bleu_stats(hyps, refs):
    from sacrebleu.metrics import BLEU
    b = BLEU()
    return b, np.array(b._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)


def paired_bootstrap_bleu(hyps_a, hyps_b, refs, n_resamples=10000, seed=0):
    """System A vs B. Returns BLEU_A, BLEU_B, 95% CIs, and the two-sided p-value that
    A and B differ (share of resamples in which the sign of the BLEU difference flips, x2)."""
    rng = np.random.default_rng(seed)
    b, sa = _bleu_stats(hyps_a, refs)
    _, sb = _bleu_stats(hyps_b, refs)
    N = len(refs)
    score = lambda s: b._compute_score_from_stats(list(s.sum(0))).score
    da, db = np.empty(n_resamples), np.empty(n_resamples)
    for i in range(n_resamples):
        ix = rng.integers(0, N, N)
        da[i], db[i] = score(sa[ix]), score(sb[ix])
    diff = da - db
    full = score(sa) - score(sb)
    p_one = (diff <= 0).mean() if full > 0 else (diff >= 0).mean()
    return {"bleu_a": score(sa), "bleu_b": score(sb),
            "ci_a": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "ci_b": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
            "p_value": float(min(1.0, 2 * max(p_one, 1.0 / n_resamples))),   # floor = 1/n_resamples
            "n_resamples": n_resamples}


def mcnemar_exact(err_a, err_b):
    """Paired sentence-level binary error indicators. Exact two-sided McNemar test."""
    a, b = np.asarray(err_a, bool), np.asarray(err_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())     # only B wrong / only A wrong
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(min(n01, n10), n, 0.5).pvalue
    return {"only_a_wrong": n10, "only_b_wrong": n01, "p_value": float(p)}
