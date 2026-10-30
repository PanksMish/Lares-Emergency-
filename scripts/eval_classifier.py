"""Accuracy / macro-F1 / confusion matrix of the classification head on a human-verified TSV
(columns: id, de, label). The category is read from the FINAL decoder state of the model's own
translation, so it is available only after the sentence is complete."""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from lares.data import Tok, BOS
from lares.decoding import offline_translate
from lares.lexicon import CATEGORIES
from lares.model import LARES

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data", default="data/processed")
ap.add_argument("--tsv", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--train_counts", help="data_stats.json (for the natural-prior precision)")
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu"); cfg = ck["cfg"]; M = cfg["model"]
tok = Tok(f"{a.data}/bpe.model")
model = LARES(tok.vocab_size, M["d_model"], M["d_ff"], M["heads"], M["enc_layers"], M["dec_layers"], M["dropout"], 5,
              M["chunk_tokens"], cfg["train"]["chunk_encoder"]); model.load_state_dict(ck["model"]); model.eval()
rows = [l.rstrip("\n").split("\t") for l in open(a.tsv, encoding="utf-8")][1:]
y, p = [], []
with torch.no_grad():
    for _, s, lab in rows:
        x = tok.encode(s)[:cfg["data"]["max_len"]]
        h, _ = offline_translate(model, x)
        src = torch.tensor([x]); mem = model.encode(src)
        tin = torch.tensor([[BOS] + h])
        hid, _ = model.decode(tin, mem, src)
        p.append(int(model.cls(model.final_state(hid, tin)).argmax())); y.append(CATEGORIES.index(lab))
cm = confusion_matrix(y, p, labels=range(len(CATEGORIES)))
res = {"n": len(y), "accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro", labels=range(5)),
       "confusion_matrix": cm.tolist(), "labels": CATEGORIES}
if a.train_counts:                                  # precision of each class at the natural class prior
    cnt = json.load(open(a.train_counts))["label_counts_train"]
    prior = np.array([cnt.get(c, 0) for c in CATEGORIES], float); prior /= prior.sum()
    support = cm.sum(1, keepdims=True)
    rate = np.divide(cm, support, out=np.zeros(cm.shape, float), where=support > 0)   # P(pred | true)
    res["precision_at_natural_prior"] = {}
    for k, c in enumerate(CATEGORIES):             # classes with no verified test examples are reported as null
        denom = prior @ rate[:, k]
        res["precision_at_natural_prior"][c] = float(prior[k] * rate[k, k] / denom) if support[k, 0] > 0 and denom > 0 else None
    res["classes_without_test_examples"] = [c for k, c in enumerate(CATEGORIES) if support[k, 0] == 0]
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
