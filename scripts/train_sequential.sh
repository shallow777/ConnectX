#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/ConnectX/bin/python
WD=/root/autodl-tmp/ConnectX_new
cd "$WD"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p logs runs results

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/train_all.log; }

run_stage() {
  local name="$1"
  shift
  log "START $name"
  "$@" 2>&1 | tee "logs/${name}.log"
  log "DONE $name"
}

# Q-learning already finished; skip if checkpoint exists.
if [[ ! -f runs/q_learning/q_learning.pkl ]]; then
  run_stage q_learning_train "$PY" -m connectx.training.train_q_learning \
    --run-dir runs/q_learning --episodes 20000
else
  log "SKIP q_learning (checkpoint exists)"
fi

PPO_RESUME=""
if [[ -f runs/ppo/checkpoints/ppo_500000.zip ]]; then
  log "SKIP ppo (ppo_500000.zip exists)"
elif compgen -G "runs/ppo/checkpoints/ppo_*.zip" > /dev/null; then
  PPO_RESUME="--resume $(ls -1 runs/ppo/checkpoints/ppo_*.zip | sort -V | tail -1)"
  run_stage ppo_train "$PY" -m connectx.training.train_ppo \
    --run-dir runs/ppo \
    --total-timesteps 500000 \
    --checkpoint-freq 50000 \
    --eval-freq 10000 \
    --negamax-games 20 \
    --add-checkpoints-to-pool \
    $PPO_RESUME
else
  run_stage ppo_train "$PY" -m connectx.training.train_ppo \
    --run-dir runs/ppo \
    --total-timesteps 500000 \
    --checkpoint-freq 50000 \
    --eval-freq 10000 \
    --negamax-games 20 \
    --add-checkpoints-to-pool
fi

if [[ -f runs/dqn/dqn.pt ]]; then
  log "SKIP dqn (dqn.pt exists)"
else
  run_stage dqn_train "$PY" -m connectx.training.train_dqn \
    --run-dir runs/dqn \
    --device cuda \
    --episodes 20000
fi

if [[ -f runs/alphazero/checkpoints/alphazero_final.pt ]] && [[ $(wc -l < runs/alphazero/negamax_curve.csv 2>/dev/null || echo 0) -ge 30 ]]; then
  log "SKIP alphazero (final checkpoint exists)"
else
  AZ_RESUME=""
  if [[ -f runs/alphazero/checkpoints/alphazero_final.pt ]]; then
    AZ_RESUME="--resume-checkpoint runs/alphazero/checkpoints/alphazero_final.pt --resume-buffer"
  elif compgen -G "runs/alphazero/checkpoints/generation_*_accepted.pt" > /dev/null; then
    AZ_RESUME="--resume-checkpoint $(ls -1 runs/alphazero/checkpoints/generation_*_accepted.pt | sort -V | tail -1) --resume-buffer"
  fi
  run_stage alphazero_train "$PY" -m connectx.training.train_alphazero \
    --run-dir runs/alphazero \
    --device cuda \
    --generations 30 \
    --selfplay-games 40 \
    --workers 2 \
    --mcts-simulations 100 \
    --eval-mcts-simulations 100 \
    --train-steps 300 \
    --arena-games 40 \
    --negamax-games 20 \
    $AZ_RESUME
fi

log "START finalize"
"$PY" -m connectx.training.finalize_results \
  --results-dir results \
  --arena-games 100 \
  2>&1 | tee logs/finalize.log

log "ALL STAGES COMPLETE"

# Chain overnight optimization (uses remaining server time).
if [[ ! -f logs/overnight_complete.marker ]]; then
  log "START overnight (chained)"
  bash scripts/train_overnight.sh 2>&1 | tee -a logs/overnight.log
  touch logs/overnight_complete.marker
  log "OVERNIGHT COMPLETE"
fi
