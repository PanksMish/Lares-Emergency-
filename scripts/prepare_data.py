"""Prepare WIT-TED German->English:  split 90/5/5, train shared BPE, truecaser, proxy labels,
and the lexicon-selected domain-adaptation set.

Input: two line-aligned plain-text files (one sentence per line). WIT3/IWSLT data must be
obtained from https://wit3.fbk.eu/ under its own licence; it is NOT redistributed here.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lares.data import read_lines, split_indices, train_spm
from lares.lexicon import label_sentence
from lares.truecase import train_truecaser

ap = argparse.ArgumentParser()
ap.add_argument("--de", required=True); ap.add_argument("--en", required=True)
ap.add_argument("--out", default="data/processed"); ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--vocab", type=int, default=30000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
de, en = read_lines(a.de), read_lines(a.en)
assert len(de) == len(en), "source/target must be line-aligned"
sp = split_indices(len(de), a.seed)
json.dump(sp, open(f"{a.out}/split_indices.json", "w"))
for name, ix in sp.items():
    open(f"{a.out}/{name}.de", "w", encoding="utf-8").write("\n".join(de[i] for i in ix) + "\n")
    open(f"{a.out}/{name}.en", "w", encoding="utf-8").write("\n".join(en[i] for i in ix) + "\n")   # cased refs
tr = sp["train"]
train_spm([de[i].lower() for i in tr] + [en[i].lower() for i in tr], f"{a.out}/bpe", a.vocab)
train_truecaser([en[i] for i in tr], f"{a.out}/truecase.model")
labels = [label_sentence(de[i].lower()) for i in tr]
adapt = [i for i, l in zip(tr, labels) if l != "Neutral"]            # lexicon-selected pairs
json.dump(adapt, open(f"{a.out}/adapt_ids.json", "w"))
stats = {"n_total": len(de), **{k: len(v) for k, v in sp.items()}, "n_adapt": len(adapt),
         "label_counts_train": {c: labels.count(c) for c in sorted(set(labels))}}
json.dump(stats, open(f"{a.out}/data_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
