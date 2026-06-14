#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/ConnectX/bin/python
WD=/root/autodl-tmp/ConnectX_new
cd "$WD"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p logs runs results/shaped submission

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/train_shaped.log; }

record_stage() {
  local stage="$1"
  local status="$2"
  local run_dir="$3"
  "$PY" -m connectx.training.record_shaped_stage \
    --stage "$stage" \
    --status "$status" \
    --run-dir "$run_dir" \
    --journal results/shaped/training_journal.csv
}

run_stage() {
  local name="$1"
  local run_dir="$2"
  shift 2
  record_stage "$name" started "$run_dir"
  log "START $name"
  if "$@" 2>&1 | tee "logs/${name}.log"; then
    record_stage "$name" completed "$run_dir"
    log "DONE $name"
  else
    record_stage "$name" failed "$run_dir"
    log "FAILED $name"
    return 1
  fi
}

run_stage q_learning_shaped runs/q_learning_shaped "$PY" -m connectx.training.train_q_learning \
  --run-dir runs/q_learning_shaped \
  --episodes 20000 \
  --reward-shaping

run_stage dqn_shaped runs/dqn_shaped "$PY" -m connectx.training.train_dqn \
  --run-dir runs/dqn_shaped \
  --device cuda \
  --episodes 20000 \
  --reward-shaping

run_stage ppo_shaped runs/ppo_shaped "$PY" -m connectx.training.train_ppo \
  --run-dir runs/ppo_shaped \
  --total-timesteps 500000 \
  --checkpoint-freq 50000 \
  --eval-freq 10000 \
  --negamax-games 20 \
  --add-checkpoints-to-pool \
  --reward-shaping

run_stage alphazero_shaped runs/alphazero_shaped "$PY" -m connectx.training.train_alphazero \
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
  --reward-shaping

record_stage export_shaped started submission
log "START export_shaped"
"$PY" -m connectx.submission.make_all_submissions \
  --output-dir submission \
  --q-learning-model runs/q_learning_shaped/q_learning.pkl \
  --dqn-model runs/dqn_shaped/dqn.pt \
  --ppo-model runs/ppo_shaped/checkpoints/ppo_500000.zip \
  --alphazero-checkpoint runs/alphazero_shaped/checkpoints/alphazero_final.pt \
  --tag shaped \
  --notes "Reward-shaping training run" \
  --validate \
  2>&1 | tee logs/export_shaped.log
record_stage export_shaped completed submission
log "DONE export_shaped"

record_stage finalize_shaped started results/shaped
log "START finalize_shaped"
"$PY" -m connectx.training.finalize_shaped_results \
  --results-dir results/shaped \
  --q-learning-model runs/q_learning_shaped/q_learning.pkl \
  --dqn-model runs/dqn_shaped/dqn.pt \
  --alphazero-checkpoint runs/alphazero_shaped/checkpoints/alphazero_final.pt \
  2>&1 | tee logs/finalize_shaped.log
log "DONE finalize_shaped"

log "SHAPED TRAINING COMPLETE"
