"""Evaluate a checkpoint on a test set (WIT-TED test split or the emergency-domain TSV).

Writes: hyps.jsonl (per-sentence outputs, delays, timings) and metrics.json.
Modes: streaming (LARES policy), waitk, offline.
"""
import argparse, json, os, platform, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lares import __version__
from lares.data import Tok, read_lines
from lares.decoding import stream_translate, waitk_translate, offline_translate
from lares.metrics import average_lagging, average_proportion, computation_aware_al, flickering_rate, quality
from lares.model import LARES
from lares.truecase import apply_truecaser
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data", default="data/processed")
ap.add_argument("--src"); ap.add_argument("--ref", help="cased references; default: <data>/test.{de,en}")
ap.add_argument("--tsv", help="emergency-domain TSV with columns id, de, en (see data/emergency/README.md)")
ap.add_argument("--mode", choices=["streaming", "waitk", "offline"], default="streaming")
ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--comet", default=None, help="e.g. Unbabel/wmt22-comet-da")
ap.add_argument("--runs", type=int, default=1, help="repeat decoding to average latency (paper: 200 on the A72)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu"); cfg = ck["cfg"]; M, Dc = cfg["model"], cfg["decode"]
tok = Tok(f"{a.data}/bpe.model")
model = LARES(tok.vocab_size, M["d_model"], M["d_ff"], M["heads"], M["enc_layers"], M["dec_layers"], M["dropout"],
              5, M["chunk_tokens"], cfg["train"]["chunk_encoder"])
model.load_state_dict(ck["model"]); model.eval()
if a.tsv:
    rows = [l.rstrip("\n").split("\t") for l in open(a.tsv, encoding="utf-8")][1:]
    ids, src, ref = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
else:
    src = read_lines(a.src or f"{a.data}/test.de"); ref = read_lines(a.ref or f"{a.data}/test.en")
    ids = list(range(len(src)))
if a.limit: ids, src, ref = ids[:a.limit], src[:a.limit], ref[:a.limit]

recs, hyps_raw = [], []
for i, s in enumerate(src):
    x = tok.encode(s)[:cfg["data"]["max_len"]]
    comp = []
    for _ in range(a.runs):
        if a.mode == "offline":
            h, ms = offline_translate(model, x); r = None
        elif a.mode == "waitk":
            r = waitk_translate(model, x, Dc["k"], Dc["src_ms"]); h, ms = r.tokens, r.compute_ms
        else:
            r = stream_translate(model, x, Dc["chunk"], Dc["tau"], Dc["src_ms"]); h, ms = r.tokens, r.compute_ms
        comp.append(ms)
    hyp = tok.decode(h); hyps_raw.append(hyp)
    n, m = len(x), len(tok.encode(ref[i]))
    rec = {"id": ids[i], "src": s, "ref": ref[i], "hyp_raw": hyp, "n_src": n, "n_hyp": len(h), "compute_ms": float(np.mean(comp))}
    if r is not None:
        rec.update(delays=r.delays, emit_ms=r.emit_ms, AL=average_lagging(r.delays, n, m),
                   AP=average_proportion(r.delays, n), AL_CA=computation_aware_al(r.emit_ms, r.delays, n, m, Dc["src_ms"]),
                   displays=r.displays)
    recs.append(rec)

hyps = apply_truecaser(f"{a.data}/truecase.model", hyps_raw)
for r, h in zip(recs, hyps): r["hyp"] = h
res = {"mode": a.mode, "n_sentences": len(recs), "n_params": model.n_params(),
       "size_mb_fp16": model.n_params() * 2 / 1e6, "size_mb_fp32": model.n_params() * 4 / 1e6,
       "latency_ms_mean": float(np.mean([r["compute_ms"] for r in recs])),
       "latency_ms_std": float(np.std([r["compute_ms"] for r in recs])),
       **quality(hyps, ref, a.comet)}
if a.mode != "offline":
    for k in ("AL", "AP", "AL_CA"):
        res[k] = float(np.nanmean([r[k] for r in recs]))
    res["FR"] = float(np.mean([flickering_rate(r["displays"]) for r in recs]))
res["env"] = {"python": platform.python_version(), "torch": torch.__version__, "lares": __version__,
              "device": "cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
              "platform": platform.platform(), "ckpt": a.ckpt, "decode_cfg": Dc}
with open(f"{a.out}/hyps.jsonl", "w", encoding="utf-8") as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(res, open(f"{a.out}/metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "env"}, indent=1))
