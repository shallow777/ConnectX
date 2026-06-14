#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/ConnectX/bin/python
WD=/root/autodl-tmp/ConnectX_new
cd "$WD"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p logs runs submission results/shaped

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/train_alphazero_shaped.log; }

record_stage() {
  "$PY" -m connectx.training.record_shaped_stage \
    --stage "$1" --status "$2" --run-dir "$3" \
    --journal results/shaped/training_journal.csv
}

log "START alphazero_shaped"
record_stage alphazero_shaped started runs/alphazero_shaped
"$PY" -m connectx.training.train_alphazero \
  --run-dir runs/alphazero_shaped \
  --device cuda \
  --generations 30 \
  --selfplay-games 40 \
  --workers 2 \
  --mcts-simulations 100 \
  --eval-mcts-simulations 100 \
  --train-steps 300 \
  --arena-games 40 \
  --negamax-games 20 \
  --reward-shaping \
  2>&1 | tee logs/alphazero_shaped.log
record_stage alphazero_shaped completed runs/alphazero_shaped
log "DONE alphazero_shaped"

log "START export_shaped"
record_stage export_shaped started submission
"$PY" -m connectx.submission.make_all_submissions \
  --output-dir submission \
  --q-learning-model runs/q_learning_shaped/q_learning.pkl \
  --dqn-model runs/dqn_shaped/dqn.pt \
  --ppo-model runs/ppo_shaped/checkpoints/ppo_500000.zip \
  --alphazero-checkpoint runs/alphazero_shaped/checkpoints/alphazero_final.pt \
  --validate \
  2>&1 | tee logs/export_shaped.log
record_stage export_shaped completed submission
log "DONE export_shaped"

log "START finalize_shaped"
record_stage finalize_shaped started results/shaped
"$PY" -m connectx.training.finalize_shaped_results \
  --results-dir results/shaped \
  2>&1 | tee logs/finalize_shaped.log
record_stage finalize_shaped completed results/shaped
log "DONE alphazero_shaped pipeline"
