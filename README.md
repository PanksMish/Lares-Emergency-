# LARES — code for low-latency German→English translation with emergency-category classification

Reference implementation (PyTorch) of **LARES: Sequence-to-Sequence Models for Low-Latency Machine Translation in Emergency Communications**
(Chandne, Mishra, Jadhav, Kadge, Okuhara).

## Status — read this first

This repository contains **code, configuration, unit tests and file-format templates only.**

It does **not** contain: trained models, training/evaluation logs, evaluation outputs, data splits from real runs,
the WIT3 corpus, or the emergency-domain evaluation sentences. These are not shipped here.
Several numbers in the revised manuscript (e.g. AP, TER, METEOR, some COMET values, AL_CA, the offline row, Table 6 tests,
error rates, classifier metrics, the leave-one-out ablation) are **provisional estimates** until reproduced by real runs;
see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Do not treat the manuscript tables as outputs of this code until
that has been done and the tables regenerated from `results/`.

Known differences to reconcile before release: the manuscript (Table 1) states beam 4, the code decodes greedily
(`decode.beam: 1`); `lares/lexicon.py` is a reference term list and must be replaced by the exact lexicon used in the paper;
`configs/lares.yaml` entries marked `(CHECK)` are not confirmed.

## What is implemented

* `lares/model.py` — chunk-causal incremental encoder (3+3 layers, d=512), classification head on the final decoder state, soft read/write policy.
* `lares/losses.py` — L = L_MT + β·L_CLS + λ·L_latency.
* `lares/decoding.py` — prefix-constrained streaming decoding (LCP commit, forced WRITE at τ), wait-k, offline, chunk wrapper for offline baselines.
* `lares/metrics.py`, `lares/stats.py` — AL, AP, FR, AL_CA, BLEU/TER/chrF/METEOR/(COMET), paired bootstrap, exact McNemar.
* `scripts/` — `prepare_data`, `train` (pretrain/adapt), `evaluate`, `eval_classifier`, `error_analysis`, `stats_compare`, `make_tables`, `run_all.sh`.

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                   # 11 tests, CPU, ~10 s
# obtain WIT3 de-en yourself (https://wit3.fbk.eu/); it is not redistributed here
bash scripts/run_all.sh train.de train.en [emergency.tsv]
```

Step by step: see `scripts/run_all.sh`. Hyper-parameters can be overridden with `--set train.lr=1e-4 model.d_model=256`.
All scripts write the configuration and environment they used next to their outputs.

## Layout

```
lares/ configs/ scripts/ tests/      code, config, tests
data/emergency/                      header-only templates + format description (no sentences)
baselines/                           how to evaluate baselines under the same protocol
results/                             empty by design
docs/REPRODUCIBILITY.md              manuscript table/figure → script mapping and status
```

## Citation / licence

See `CITATION.cff`. Code: MIT (authors may change before release). WIT3/IWSLT data is under its own licence.
