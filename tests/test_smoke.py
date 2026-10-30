"""Smoke / unit tests on toy data (no WIT-TED needed). These check that the code is wired
correctly; they say NOTHING about translation quality on real data."""
import math, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lares.data import Example, PrefixMixDataset, collate, split_indices, PAD, BOS, EOS
from lares.decoding import stream_translate, waitk_translate, chunk_wrapper_stream, lcp
from lares.lexicon import label_sentence
from lares.losses import total_loss
from lares.metrics import average_lagging, average_proportion, flickering_rate, computation_aware_al
from lares.model import LARES
from lares.stats import mcnemar_exact, paired_bootstrap_bleu

V = 40


def tiny():
    return LARES(V, d_model=32, d_ff=64, heads=4, enc_layers=1, dec_layers=1, dropout=0.0, chunk_tokens=2, max_len=64)


def test_al_ap_hand_computed():
    # wait-3, n=m=6: g = 3,4,5,6,6,6, T*=4  ->  AL = mean(3-0, 4-1, 5-2, 6-3) = 3
    assert average_lagging([3, 4, 5, 6, 6, 6], 6, 6) == 3.0
    assert math.isclose(average_proportion([3, 4, 5, 6, 6, 6], 6), (3 + 4 + 5 + 6 + 6 + 6) / 36)
    # fully offline: AL equals the source length
    assert average_lagging([6] * 6, 6, 6) == 6.0
    # AP of any non-anticipating policy is >= 1/2 + something; never below 1/m * ... sanity: AP>0.5 for wait-3
    assert average_proportion([3, 4, 5, 6, 6, 6], 6) > 0.5


def test_flickering_rate():
    assert flickering_rate([[1], [1, 2], [1, 2, 3]]) == 0.0
    assert flickering_rate([[1, 2], [1, 3]]) > 0.0          # token 2 revised


def test_split_is_deterministic_and_disjoint():
    a, b = split_indices(1000, 13), split_indices(1000, 13)
    assert a == b
    s = [set(a[k]) for k in ("train", "valid", "test")]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
    assert (len(a["train"]), len(a["valid"]), len(a["test"])) == (900, 50, 50)


def test_prefix_mix_ratio_and_rho():
    ex = [Example(list(range(4, 14)), list(range(4, 14)), 0, i) for i in range(1)]
    ds = PrefixMixDataset(ex, alpha=0.7, seed=1)
    full = sum(len(ds[0].src) == 10 for _ in range(4000)) / 4000
    assert 0.65 < full < 0.78                                # ~alpha (k=n counts as 'full' too)
    for _ in range(200):
        e = ds[0]
        assert len(e.tgt) == max(1, (len(e.src) * 10) // 10)


def test_lexicon_priority():
    assert label_sentence("Feuer im Krankenhaus") == "Medical"      # Medical > Fire
    assert label_sentence("Der Himmel ist blau") == "Neutral"


def test_forward_backward_and_losses():
    m = tiny(); r = random.Random(0)
    batch = collate([Example([r.randint(4, V - 1) for _ in range(7)], [r.randint(4, V - 1) for _ in range(6)], 1)
                     for _ in range(3)])
    out = m(batch["src"], batch["tgt_in"], want_policy=True)
    assert out["g_hat"].shape == batch["tgt_in"].shape
    assert (out["g_hat"] >= 1 - 1e-5).all() and (out["g_hat"] <= batch["src"].size(1) + 1e-5).all()
    loss, parts = total_loss(out, batch, {"beta": 0.2, "lambda": 0.05, "joint_cls": True, "latency_loss": True})
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert set(parts) == {"mt", "cls", "lat", "total"}


def test_chunk_encoder_is_incremental():
    """Encoding a prefix must give the same states as the prefix part of encoding the full source."""
    m = tiny().eval(); x = torch.randint(4, V, (1, 8))
    with torch.no_grad():
        full, pre = m.encode(x), m.encode(x[:, :4])
    assert torch.allclose(full[:, :4], pre, atol=1e-5)


def test_learns_copy_task_and_streams_without_revision():
    torch.manual_seed(0); random.seed(0)
    m = tiny(); opt = torch.optim.Adam(m.parameters(), 3e-3)
    for step in range(400):
        ex = []
        for _ in range(32):
            L = random.randint(4, 8); x = [random.randint(4, 12) for _ in range(L)]
            ex.append(Example(x, x, 0))
        b = collate(ex)
        out = m(b["src"], b["tgt_in"], want_policy=True)
        loss, _ = total_loss(out, b, {"beta": 0.0, "lambda": 0.0, "joint_cls": False, "latency_loss": False, "label_smoothing": 0.0})
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss.detach()) < 0.3, "toy copy task did not train"
    x = [5, 6, 7, 8, 9, 10]
    r = stream_translate(m, x, chunk=2, tau=3)
    assert len(r.tokens) == len(r.delays) == len(r.emit_ms)
    assert r.delays == sorted(r.delays)                       # delays never decrease
    assert r.delays[-1] <= len(x) and all(d >= 1 for d in r.delays)
    assert r.tokens == x                                      # copy task is solved by the streaming decoder
    w = waitk_translate(m, x, k=3)
    assert w.delays[0] == 3 and w.tokens == x
    al = average_lagging(r.delays, len(x), len(r.tokens))
    assert al < len(x)                                         # streaming lags less than offline
    assert computation_aware_al(r.emit_ms, r.delays, len(x), len(r.tokens)) > 0


def test_committed_prefix_is_never_revised():
    """Instrument greedy_extend: every call must start from the committed prefix."""
    import lares.decoding as D
    m = tiny().eval(); seen = []
    orig = D.greedy_extend
    def spy(model, src, forced, max_new):
        out = orig(model, src, forced, max_new); seen.append((list(forced), out)); return out
    D.greedy_extend = spy
    try:
        r = D.stream_translate(m, [5, 6, 7, 8, 9, 10, 11, 12], chunk=2, tau=2)
    finally:
        D.greedy_extend = orig
    for forced, out in seen:
        assert out[:len(forced)] == forced                     # forced prefix is preserved verbatim
    assert r.tokens[:len(seen[-1][0])] == seen[-1][0]


def test_chunk_wrapper_records_displays():
    outs = {2: [1], 4: [2, 3], 6: [2, 3, 4]}
    r = chunk_wrapper_stream(lambda s: outs[len(s)], [0] * 6, chunk=2)
    assert r.tokens == [1, 3, 4] or r.tokens == [1, 3, 4][:len(r.tokens)]
    assert len(r.displays) == 3 and flickering_rate(r.displays) > 0   # the wrapper revised token 1 -> 2


def test_stats():
    assert mcnemar_exact([1, 0, 1, 1], [1, 0, 1, 1])["p_value"] == 1.0
    a = [1] * 30 + [0] * 5; b = [0] * 30 + [0] * 5
    assert mcnemar_exact(a, b)["p_value"] < 1e-6
    hyp = ["the cat sat on the mat", "a dog barks loudly", "he reads a book", "it rains today"] * 10
    ref = hyp[:]
    res = paired_bootstrap_bleu(hyp, hyp, ref, n_resamples=200)
    assert res["p_value"] == 1.0 and res["bleu_a"] == res["bleu_b"]
