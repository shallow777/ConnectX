#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/ConnectX/bin/python
WD=/root/autodl-tmp/ConnectX_new
cd "$WD"
export PYTHONPATH=.

echo "[$(date '+%F %T')] Waiting for AlphaZero overnight..."
while pgrep -f "[p]ython -m connectx.training.train_alphazero.*alphazero_overnight" >/dev/null; do
  gens=$(wc -l < runs/alphazero_overnight/negamax_curve.csv 2>/dev/null || echo 0)
  echo "[$(date '+%F %T')] still running, curve lines=$gens"
  sleep 120
done

echo "[$(date '+%F %T')] Exporting all four submissions..."
$PY -m connectx.submission.make_all_submissions \
  --output-dir submission \
  --validate \
  --tag overnight_gen38 \
  --notes "Best overnight negamax ~90%; MCTS budget 88%; multi-threat tactical" \
  2>&1 | tee logs/make_all_submissions.log

touch logs/overnight_complete.marker
echo "[$(date '+%F %T')] Done."
