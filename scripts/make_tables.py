"""Build a LaTeX table body from metrics.json files. No numbers are hard-coded: every cell comes from a file.
usage: make_tables.py NAME=path/to/metrics.json [NAME=...] > table.tex"""
import json, sys
COLS = [("bleu", "BLEU", 1), ("ter", "TER", 1), ("chrf", "chrF", 1), ("meteor", "METEOR", 1), ("comet", "COMET", 3),
        ("AL", "AL", 2), ("AP", "AP", 2), ("AL_CA", "AL$_{CA}$", 0), ("FR", "FR", 3), ("latency_ms_mean", "Lat.", 1)]
print(" & ".join(["System"] + [c[1] for c in COLS]) + r" \\")
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    m = json.load(open(path))
    cells = []
    for key, _, nd in COLS:
        v = m.get(key)
        cells.append("--" if v is None or v != v else f"{v:.{nd}f}")
    print(" & ".join([name] + cells) + r" \\")
