#!/usr/bin/env bash
# 过夜强化训练 (overnight optimization): 在主流水线之后追加算力。
# Phase 1: PPO 续训到 150 万步; Phase 2: 更强设置的 AlphaZero (单独 run 目录,
# 从主训练最优 checkpoint 热启动); Phase 3: 重新汇总结果并生成 submission。
# 在仓库根目录运行: bash scripts/train_overnight.sh
set -euo pipefail

PY=${PY:-python}
cd "$(dirname "$0")/.."
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

# 取主训练里最新被接受的 candidate 作为热启动种子 (warm-start seed)
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

# --- Phase 1: PPO 续训 (+1M steps, 对手池更深) ---
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

# --- Phase 2: 更强 AlphaZero (更多 generation / simulation, 阈值略放宽) ---
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
  # 把主训练的 replay buffer 复制过来继续用, 减少冷启动样本浪费
  if [[ -f runs/alphazero/replay_buffer.npz ]]; then
    mkdir -p runs/alphazero_overnight
    cp -f runs/alphazero/replay_buffer.npz runs/alphazero_overnight/replay_buffer.npz
    AZ_ARGS+=(--resume-buffer)
  fi
  if [[ -f runs/alphazero_overnight/checkpoints/alphazero_final.pt ]] \
     && [[ $(wc -l < runs/alphazero_overnight/negamax_curve.csv 2>/dev/null || echo 0) -ge 61 ]]; then
    log "SKIP alphazero_overnight (already complete)"
  else
    # 已有 final checkpoint 时改为从它续训
    if [[ -f runs/alphazero_overnight/checkpoints/alphazero_final.pt ]]; then
      AZ_ARGS=(--run-dir runs/alphazero_overnight --device cuda
        --resume-checkpoint runs/alphazero_overnight/checkpoints/alphazero_final.pt --resume-buffer
        --generations 60 --selfplay-games 60 --workers 2
        --mcts-simulations 150 --eval-mcts-simulations 120 --train-steps 500 --batch-size 256
        --arena-games 60 --accept-threshold 0.52 --negamax-games 30)
    fi
    run_stage alphazero_overnight "$PY" -m connectx.training.train_alphazero "${AZ_ARGS[@]}"
  fi
else
  log "SKIP alphazero_overnight (no seed checkpoint)"
fi

# --- Phase 3: 用所有 run 里的最优 checkpoint 重新汇总 + 生成 submission ---
run_stage finalize_overnight "$PY" -m connectx.training.finalize_results \
  --results-dir results \
  --arena-games 200 \
  --alphazero-simulations 120

log "=== OVERNIGHT OPTIMIZATION COMPLETE ==="
