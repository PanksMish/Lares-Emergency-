#!/usr/bin/env bash
# Full pipeline on YOUR copy of WIT3 de-en. Nothing here ships data or results.
# usage: bash scripts/run_all.sh path/to/train.de path/to/train.en [emergency.tsv]
set -euo pipefail
DE=$1; EN=$2; EMG=${3:-}
python scripts/prepare_data.py --de "$DE" --en "$EN" --out data/processed
python scripts/train.py --out runs/pretrain
python scripts/train.py --out runs/adapt --stage adapt --init runs/pretrain/model.pt
for m in streaming waitk offline; do
  python scripts/evaluate.py --ckpt runs/adapt/model.pt --mode $m --out results/test_$m
done
python scripts/stats_compare.py --a results/test_streaming/hyps.jsonl --b results/test_offline/hyps.jsonl --out results/cmp_stream_vs_offline.json
if [ -n "$EMG" ]; then
  python scripts/evaluate.py --ckpt runs/adapt/model.pt --tsv "$EMG" --mode streaming --out results/emergency_streaming
fi
