"""Train LARES (stage 'pretrain') or fine-tune on the lexicon-selected set (stage 'adapt')."""
import argparse, json, os, random, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lares.data import Tok, build_examples, read_lines, PrefixMixDataset, collate
from lares.losses import total_loss
from lares.model import LARES
from lares.decoding import offline_translate
from lares.truecase import apply_truecaser
from lares.metrics import quality
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/lares.yaml"); ap.add_argument("--out", required=True)
ap.add_argument("--stage", choices=["pretrain", "adapt"], default="pretrain")
ap.add_argument("--init", help="checkpoint to start from (required for --stage adapt)")
ap.add_argument("--set", nargs="*", default=[], help="override, e.g. train.joint_cls=false")
ap.add_argument("--max_steps", type=int, default=0, help="debug: stop early")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
for kv in a.set:
    k, v = kv.split("="); d = cfg
    *path, last = k.split(".")
    for p in path: d = d[p]
    d[last] = yaml.safe_load(v)
os.makedirs(a.out, exist_ok=True)
yaml.safe_dump(cfg, open(f"{a.out}/config.used.yaml", "w"))
random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
D, M, T = cfg["data"], cfg["model"], cfg["train"]
tok = Tok(f"{D['dir']}/bpe.model")
tr = build_examples(read_lines(f"{D['dir']}/train.de"), read_lines(f"{D['dir']}/train.en"), tok, D["max_len"])
if a.stage == "adapt":
    ids = set(json.load(open(f"{D['dir']}/adapt_ids.json")))
    sp = json.load(open(f"{D['dir']}/split_indices.json"))["train"]
    keep = {j for j, i in enumerate(sp) if i in ids}
    tr = [e for e in tr if e.idx in keep]
    print(f"adaptation set: {len(tr)} pairs")
va_src, va_ref = read_lines(f"{D['dir']}/valid.de"), read_lines(f"{D['dir']}/valid.en")
model = LARES(tok.vocab_size, M["d_model"], M["d_ff"], M["heads"], M["enc_layers"], M["dec_layers"],
              M["dropout"], 5, M["chunk_tokens"], T["chunk_encoder"]).to(dev)
if a.init:
    model.load_state_dict(torch.load(a.init, map_location=dev)["model"])
print("parameters:", model.n_params())
lr = cfg["adapt"]["lr"] if a.stage == "adapt" else T["lr"]
epochs = cfg["adapt"]["epochs"] if a.stage == "adapt" else T["epochs"]
opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
warm = 0 if a.stage == "adapt" else T["warmup_steps"]
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) if warm else 1.0)
scaler = torch.amp.GradScaler(enabled=(T["fp16"] and dev == "cuda"))
log = open(f"{a.out}/train_log.jsonl", "a")


def batches(ds):
    """Token-bucketed batches of ~batch_tokens target+source tokens."""
    order = sorted(range(len(ds)), key=lambda i: len(ds.ex[i].src) + random.random())
    cur, ntok, out = [], 0, []
    for i in order:
        l = len(ds.ex[i].src) + len(ds.ex[i].tgt)
        if cur and ntok + l > T["batch_tokens"]:
            out.append(cur); cur, ntok = [], 0
        cur.append(i); ntok += l
    if cur: out.append(cur)
    random.shuffle(out)
    return out


def validate():
    hyps = []
    for s in va_src[:500]:
        h, _ = offline_translate(model, tok.encode(s)[:D["max_len"]], device=dev)
        hyps.append(tok.decode(h))
    hyps = apply_truecaser(f"{D['dir']}/truecase.model", hyps)
    return quality(hyps, va_ref[:500])["bleu"]


best, bad, step = -1.0, 0, 0
for ep in range(1, epochs + 1):
    ds = PrefixMixDataset(tr, T["alpha"], T["prefix_aug"], seed=cfg["seed"] + ep)
    model.train()
    for ix in batches(ds):
        b = {k: v.to(dev) for k, v in collate([ds[i] for i in ix]).items()}
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(T["fp16"] and dev == "cuda")):
            out = model(b["src"], b["tgt_in"], want_policy=T["latency_loss"])
            loss, parts = total_loss(out, b, {**T, "lambda": T["lambda"]})
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        if step % 100 == 0:
            log.write(json.dumps({"step": step, "epoch": ep, **{k: float(v) for k, v in parts.items()}}) + "\n"); log.flush()
        if a.max_steps and step >= a.max_steps: break
    bleu = validate() if va_src and not a.max_steps else float("nan")
    log.write(json.dumps({"epoch": ep, "valid_bleu": bleu}) + "\n"); log.flush()
    print(f"epoch {ep} valid BLEU {bleu:.2f}")
    if a.max_steps and step >= a.max_steps: break
    if a.stage == "adapt" or bleu > best:
        best, bad = bleu, 0
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "valid_bleu": bleu}, f"{a.out}/model.pt")
    else:
        bad += 1
        if bad >= T["patience"]: break
if a.max_steps:
    torch.save({"model": model.state_dict(), "cfg": cfg}, f"{a.out}/model.pt")
