"""Manual error analysis. Input TSV (one row per sentence per system):
   id <TAB> system <TAB> entity_error(0/1) <TAB> compound_error(0/1) [<TAB> word_order <TAB> omission <TAB> hallucination]
Each rate is the share of the n annotated sentences with >=1 error of that category (multiples of 100/n %).
Paired exact McNemar tests are run between every system and the reference system."""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lares.stats import mcnemar_exact

ap = argparse.ArgumentParser()
ap.add_argument("--tsv", required=True); ap.add_argument("--ref_system", default="LARES"); ap.add_argument("--out", required=True)
a = ap.parse_args()
cats = ["entity", "compound", "word_order", "omission", "hallucination"]
rows = [l.rstrip("\n").split("\t") for l in open(a.tsv, encoding="utf-8")][1:]
data = collections.defaultdict(dict)
for r in rows:
    data[r[1]][r[0]] = [int(v) for v in r[2:]]
res = {"rates_pct": {}, "mcnemar_vs_" + a.ref_system: {}}
ref = data[a.ref_system]
for sysname, d in data.items():
    ids = sorted(d)
    res["rates_pct"][sysname] = {c: 100 * sum(d[i][k] for i in ids) / len(ids) for k, c in enumerate(cats) if len(d[ids[0]]) > k}
    if sysname != a.ref_system:
        common = sorted(set(d) & set(ref))
        res["mcnemar_vs_" + a.ref_system][sysname] = {
            c: mcnemar_exact([d[i][k] for i in common], [ref[i][k] for i in common])
            for k, c in enumerate(cats[:2])}
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
