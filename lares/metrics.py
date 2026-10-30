"""Quality and latency metrics (Sec. 2.2, 4)."""
from __future__ import annotations
import numpy as np


# ---------------- latency / streaming ----------------
def _t_star(delays, n):
    for i, g in enumerate(delays, 1):
        if g >= n:
            return i
    return len(delays)


def average_lagging(delays, n, m):
    """Eq. (4): AL = 1/T* sum_{t<=T*} (g(t) - (t-1)/r), r = m/n, T* = first t with g(t)=n.
    n = source length, m = reference length (as in SimulEval) or hypothesis length."""
    if not delays:
        return float("nan")
    T = _t_star(delays, n)
    r = m / n
    return float(np.mean([delays[t - 1] - (t - 1) / r for t in range(1, T + 1)]))


def average_proportion(delays, n):
    """Eq. (5): AP = 1/(n*m) sum_t g(t), m = number of emitted tokens."""
    return float(np.sum(delays) / (n * len(delays))) if delays else float("nan")


def computation_aware_al(emit_ms, delays, n, m, src_ms=250.0):
    """AL_CA in ms: like AL but with the (simulated) wall-clock emission time of each token,
    compute included, and an ideal emission rate of n*src_ms/m ms per target token."""
    if not emit_ms:
        return float("nan")
    T = _t_star(delays, n)
    rate = n * src_ms / m
    return float(np.mean([emit_ms[t - 1] - (t - 1) * rate for t in range(1, T + 1)]))


def flickering_rate(displays):
    """Eq. (6): revised tokens / emitted tokens over successive displayed hypotheses."""
    revised = emitted = 0
    prev = []
    for d in displays:
        k = 0
        while k < min(len(prev), len(d)) and prev[k] == d[k]:
            k += 1
        revised += len(prev) - k
        emitted += len(d) - k
        prev = d
    return revised / emitted if emitted else 0.0


# ---------------- quality ----------------
def quality(hyps, refs, comet_model=None):
    """hyps/refs: detokenised, CASED strings. SacreBLEU signature is returned for the paper."""
    import sacrebleu
    from sacrebleu.metrics import BLEU
    metric = BLEU()
    bleu = metric.corpus_score(hyps, [refs])
    out = {"bleu": bleu.score, "bleu_signature": str(metric.get_signature()),
           "ter": sacrebleu.corpus_ter(hyps, [refs]).score, "chrf": sacrebleu.corpus_chrf(hyps, [refs]).score}
    out["meteor"] = meteor(hyps, refs)
    if comet_model:
        out["comet"] = comet(comet_model, hyps, refs)
    return out


def meteor(hyps, refs):
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
        return 100.0 * float(np.mean([meteor_score([r.split()], h.split()) for h, r in zip(hyps, refs)]))
    except Exception as e:                                # keep evaluation going, but say so
        print(f"[warn] METEOR unavailable: {e}")
        return float("nan")


def comet(name, hyps, refs, srcs=None):
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(name))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs or [""] * len(hyps), hyps, refs)]
    return float(model.predict(data, batch_size=32, gpus=0).system_score)
