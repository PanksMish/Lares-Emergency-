"""Streaming decoding with prefix-constrained decoding and LCP (local-agreement) commitment.

Committed tokens are fed back as a forced decoder prefix, so they are never revised (Sec. 3.2).
READ/WRITE policy (Eq. 15): WRITE if |Y_stable| > |Y_commit| or Delta_t = |X_<=g| - |Y_commit| >= tau.
Time is *simulated*: source tokens arrive every `src_ms` ms; measured model compute time is added.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import torch

from .data import BOS, EOS


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@torch.no_grad()
def greedy_extend(model, src, forced, max_new):
    """Greedy continuation of `forced` given source `src` (1,S). Returns list WITHOUT BOS/EOS."""
    mem = model.encode(src)
    ys = [BOS] + list(forced)
    for _ in range(max_new):
        tgt = torch.tensor([ys], device=src.device)
        h, logits = model.decode(tgt, mem, src)
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        ys.append(nxt)
    return ys[1:]


@dataclass
class StreamResult:
    tokens: list
    delays: list                  # g(t): source tokens read when target token t was written
    emit_ms: list                 # simulated wall-clock emission time of each token (ms)
    compute_ms: float             # total model compute time for the sentence
    displays: list = field(default_factory=list)   # successive displayed hypotheses (for FR)


@torch.no_grad()
def stream_translate(model, src_ids, chunk=2, tau=3, src_ms=250.0, max_ratio=2.0, device="cpu"):
    model.eval()
    n = len(src_ids)
    max_new = int(max_ratio * n) + 10
    commit, prev, delays, emit, clock, comp = [], [], [], [], 0.0, 0.0
    reads = 0
    while reads < n:
        reads = min(reads + chunk, n)                                   # READ one chunk
        clock = max(clock, reads * src_ms)                              # wait for the source
        t0 = time.perf_counter()
        src = torch.tensor([src_ids[:reads]], device=device)
        hyp = greedy_extend(model, src, commit, max_new)
        dt = (time.perf_counter() - t0) * 1000.0
        clock, comp = clock + dt, comp + dt
        stable = hyp[:lcp(hyp, prev)] if prev else []
        prev = hyp
        if reads == n:
            new = hyp                                                   # flush at end of source
        elif len(stable) > len(commit):
            new = stable
        elif reads - len(commit) >= tau and len(hyp) > len(commit):     # forced WRITE (Delta_t >= tau)
            new = hyp[:len(commit) + 1]
        else:
            new = commit
        for _ in new[len(commit):]:
            delays.append(reads)
            emit.append(clock)
        commit = list(new)
    return StreamResult(commit, delays, emit, comp, displays=[list(commit)])


@torch.no_grad()
def offline_translate(model, src_ids, max_ratio=2.0, device="cpu"):
    model.eval()
    t0 = time.perf_counter()
    hyp = greedy_extend(model, torch.tensor([src_ids], device=device), [], int(max_ratio * len(src_ids)) + 10)
    return hyp, (time.perf_counter() - t0) * 1000.0


@torch.no_grad()
def waitk_translate(model, src_ids, k=3, src_ms=250.0, max_ratio=2.0, device="cpu"):
    """Wait-k with a prefix-trained model: at step t read g(t)=min(k+t-1,n) tokens, write one token."""
    model.eval()
    n, commit, delays, emit, clock, comp = len(src_ids), [], [], [], 0.0, 0.0
    for t in range(1, int(max_ratio * n) + 10):
        g = min(k + t - 1, n)
        clock = max(clock, g * src_ms)
        t0 = time.perf_counter()
        src = torch.tensor([src_ids[:g]], device=device)
        mem = model.encode(src)
        h, logits = model.decode(torch.tensor([[BOS] + commit], device=device), mem, src)
        nxt = int(logits[0, -1].argmax())
        dt = (time.perf_counter() - t0) * 1000.0
        clock, comp = clock + dt, comp + dt
        if nxt == EOS and g == n:
            break
        if nxt == EOS:                       # not allowed to stop before the source ends
            continue
        commit.append(nxt); delays.append(g); emit.append(clock)
    return StreamResult(commit, delays, emit, comp)


def chunk_wrapper_stream(translate_fn, src_ids, chunk=2, src_ms=250.0):
    """Wrap an OFFLINE system (EdgeFormer, MobileNMT, ...) in a simulated chunk-streaming harness.
    `translate_fn(list[int]) -> list[int]` re-translates the full available prefix at every chunk.
    Already written tokens are never retracted (as in SimulEval); `displays` keeps every successive
    hypothesis so the Flickering Rate (Eq. 6) can be computed."""
    n, written, delays, emit, displays, clock, comp, reads = len(src_ids), [], [], [], [], 0.0, 0.0, 0
    while reads < n:
        reads = min(reads + chunk, n)
        clock = max(clock, reads * src_ms)
        t0 = time.perf_counter()
        hyp = translate_fn(src_ids[:reads])
        dt = (time.perf_counter() - t0) * 1000.0
        clock, comp = clock + dt, comp + dt
        displays.append(list(hyp))
        for tok in hyp[len(written):]:
            written.append(tok); delays.append(reads); emit.append(clock)
    return StreamResult(written, delays, emit, comp, displays)
