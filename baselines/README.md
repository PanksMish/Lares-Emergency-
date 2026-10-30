# Baselines

No baseline code or checkpoints are bundled. Obtain each from its original authors and evaluate it under the same
protocol (same split, truecasing, SacreBLEU `case:mixed`, 250 ms/token simulated arrival).

* **Offline systems** (e.g. EdgeFormer, MobileNMT): decode with their own toolkit, then wrap them with
  `lares.decoding.chunk_wrapper_stream` to obtain streaming latency/flicker numbers. Keep `displays` for FR.
* **Wait-k / STACL**: use the original implementation (Fairseq/STACL, k=3) or `scripts/evaluate.py --mode waitk` (this repo's own simple wait-k on the LARES model).
* **DiG-SST**: text-input variant only; state in the paper exactly how it was adapted.

Record for every baseline: source/commit, hyper-parameters, parameter count, and the command that produced the hypotheses.
Write hypotheses in the same `hyps.jsonl` format as `evaluate.py` so `stats_compare.py` and `make_tables.py` work unchanged.
