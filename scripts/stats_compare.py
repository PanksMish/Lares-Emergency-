"""Paired bootstrap significance/CI between two systems' hyps.jsonl, plus optional McNemar."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lares.stats import paired_bootstrap_bleu

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--n", type=int, default=10000); ap.add_argument("--out", required=True)
a = ap.parse_args()
x = [json.loads(l) for l in open(a.a, encoding="utf-8")]
y = [json.loads(l) for l in open(a.b, encoding="utf-8")]
assert [r["id"] for r in x] == [r["id"] for r in y], "systems must be evaluated on the same ids"
res = paired_bootstrap_bleu([r["hyp"] for r in x], [r["hyp"] for r in y], [r["ref"] for r in x], a.n)
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
