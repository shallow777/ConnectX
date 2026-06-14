#!/usr/bin/env bash
# Overnight optimization: runs automatically after the main pipeline.
# Uses stronger AlphaZero settings + extended PPO self-play, then re-finalizes.
set -euo pipefail

PY=/root/miniconda3/envs/ConnectX/bin/python
WD=/root/autodl-tmp/ConnectX_new
cd "$WD"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p logs runs results

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/overnight.log; }

run_stage() {
  local name="$1"
  shift
  log "START $name"
  "$@" 2>&1 | tee -a "logs/${name}.log"
  log "DONE $name"
}

best_az_seed() {
  if compgen -G "runs/alphazero/checkpoints/generation_*_accepted.pt" > /dev/null; then
    ls -1 runs/alphazero/checkpoints/generation_*_accepted.pt | sort -V | tail -1
  elif [[ -f runs/alphazero/checkpoints/alphazero_final.pt ]]; then
    echo runs/alphazero/checkpoints/alphazero_final.pt
  else
    echo ""
  fi
}

log "=== OVERNIGHT OPTIMIZATION START ==="

# --- Phase 1: Extended PPO (+1M steps, deeper opponent pool) ---
PPO_TARGET=1500000
PPO_LATEST=$(ls -1 runs/ppo/checkpoints/ppo_*.zip 2>/dev/null | sort -V | tail -1 || true)
if [[ -n "$PPO_LATEST" ]]; then
  PPO_DONE=$(basename "$PPO_LATEST" .zip | cut -d_ -f2)
  if [[ "$PPO_DONE" -lt "$PPO_TARGET" ]]; then
    run_stage ppo_overnight "$PY" -m connectx.training.train_ppo \
      --run-dir runs/ppo \
      --resume "$PPO_LATEST" \
      --total-timesteps "$PPO_TARGET" \
      --checkpoint-freq 100000 \
      --eval-freq 20000 \
      --negamax-games 30 \
      --add-checkpoints-to-pool
  else
    log "SKIP ppo_overnight (already at ${PPO_DONE})"
  fi
else
  log "SKIP ppo_overnight (no checkpoint)"
fi

# --- Phase 2: Stronger AlphaZero (new run dir, warm-start from best champion) ---
AZ_SEED=$(best_az_seed)
if [[ -n "$AZ_SEED" ]]; then
  AZ_ARGS=(
    --run-dir runs/alphazero_overnight
    --device cuda
    --resume-checkpoint "$AZ_SEED"
    --generations 60
    --selfplay-games 60
    --workers 2
    --mcts-simulations 150
    --eval-mcts-simulations 120
    --train-steps 500
    --batch-size 256
    --arena-games 60
    --accept-threshold 0.52
    --negamax-games 30
  )
  if [[ -f runs/alphazero/replay_buffer.npz ]]; then
    mkdir -p runs/alphazero_overnight
    cp -f runs/alphazero/replay_buffer.npz runs/alphazero_overnight/replay_buffer.npz
    AZ_ARGS+=(--resume-buffer)
  fi
  if [[ -f runs/alphazero_overnight/checkpoints/alphazero_final.pt ]] \
     && [[ $(wc -l < runs/alphazero_overnight/negamax_curve.csv 2>/dev/null || echo 0) -ge 61 ]]; then
    log "SKIP alphazero_overnight (already complete)"
  elif [[ -f runs/alphazero_overnight/checkpoints/alphazero_final.pt ]]; then
    AZ_ARGS=(--run-dir runs/alphazero_overnight --device cuda --resume-checkpoint runs/alphazero_overnight/checkpoints/alphazero_final.pt --resume-buffer --generations 60 --selfplay-games 60 --workers 2 --mcts-simulations 150 --eval-mcts-simulations 120 --train-steps 500 --batch-size 256 --arena-games 60 --accept-threshold 0.52 --negamax-games 30)
    run_stage alphazero_overnight "$PY" -m connectx.training.train_alphazero "${AZ_ARGS[@]}"
  else
    run_stage alphazero_overnight "$PY" -m connectx.training.train_alphazero "${AZ_ARGS[@]}"
  fi
else
  log "SKIP alphazero_overnight (no seed checkpoint)"
fi

# --- Phase 3: Re-finalize with best checkpoints across all runs ---
run_stage finalize_overnight "$PY" -m connectx.training.finalize_results \
  --results-dir results \
  --arena-games 200 \
  --alphazero-simulations 120

log "=== OVERNIGHT OPTIMIZATION COMPLETE ==="
