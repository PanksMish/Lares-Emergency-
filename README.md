# LARES — code for low-latency German→English translation with emergency-category classification

Reference implementation (PyTorch) of **LARES: Sequence-to-Sequence Models for Low-Latency Machine Translation in Emergency Communications**
(Chandne, Mishra, Jadhav, Kadge, Okuhara).


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
