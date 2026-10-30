# Reproducibility map

Status legend: **code ready** = script exists and is smoke-tested on a toy corpus; **needs run** = no real run has been
performed with this repository, so the corresponding manuscript numbers are provisional.

| Manuscript item | Script | Status |
|---|---|---|
| Data split 90/5/5, BPE 30k, truecaser | `prepare_data.py` | code ready; needs run on WIT3 |
| Pre-training and domain adaptation | `train.py --stage pretrain/adapt` | code ready; needs run |
| Table 2 quality (BLEU/TER/chrF/METEOR/COMET) | `evaluate.py --mode offline/streaming` | code ready; needs run (TER, METEOR, some COMET provisional) |
| Table 2 latency (AL, AP, FR, AL_CA, ms) | `evaluate.py --runs N` | code ready; needs run on target hardware |
| Wait-k / chunk-wrapped baselines | `evaluate.py --mode waitk`, `baselines/` | code ready; external baselines need run |
| Significance, CIs (Table 6, Fig. 2b) | `stats_compare.py --n 10000` | code ready; needs run |
| Classifier accuracy / macro-F1 / confusion / natural-prior precision | `eval_classifier.py` | code ready; needs verified labels |
| Error analysis and McNemar | `error_analysis.py` | code ready; needs annotations |
| Ablations (Table 4/5) | `train.py --set train.joint_cls=false ...` | code ready; needs run |
| Emergency-domain evaluation | `evaluate.py --tsv` | code ready; set not included |
| Tables as LaTeX | `make_tables.py` | code ready |

Open items before any release of results: reconcile beam size (paper 4 vs code 1), confirm the lexicon, confirm
`(CHECK)` hyper-parameters, run all steps, replace every provisional manuscript number, and then publish the
`config.used.yaml`, `train_log.jsonl`, `metrics.json` and `hyps.jsonl` files of those runs.
