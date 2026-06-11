#!/usr/bin/env bash
# 主训练流水线 (main training pipeline): Q-learning -> PPO -> DQN -> AlphaZero -> finalize。
# 每个阶段如果发现已有 checkpoint 会自动跳过/续训, 因此中断后重跑是安全的 (resume-safe)。
# 在仓库根目录运行: bash scripts/train_sequential.sh
set -euo pipefail

PY=${PY:-python}                       # 可用环境变量覆盖 python 解释器
cd "$(dirname "$0")/.."                # 切到仓库根目录
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

# --- Stage 1: tabular Q-learning (小棋盘 4x5 connect-3 基线) ---
if [[ ! -f runs/q_learning/q_learning.pkl ]]; then
  run_stage q_learning_train "$PY" -m connectx.training.train_q_learning \
    --run-dir runs/q_learning --episodes 20000
else
  log "SKIP q_learning (checkpoint exists)"
fi

# --- Stage 2: MaskablePPO self-play (标准 6x7 棋盘) ---
PPO_RESUME=""
if [[ -f runs/ppo/checkpoints/ppo_500000.zip ]]; then
  log "SKIP ppo (ppo_500000.zip exists)"
elif compgen -G "runs/ppo/checkpoints/ppo_*.zip" > /dev/null; then
  # 有中间 checkpoint 时从最新的续训
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

# --- Stage 3: DQN self-play (标准 6x7 棋盘) ---
if [[ -f runs/dqn/dqn.pt ]]; then
  log "SKIP dqn (dqn.pt exists)"
else
  run_stage dqn_train "$PY" -m connectx.training.train_dqn \
    --run-dir runs/dqn \
    --device cuda \
    --episodes 20000
fi

# --- Stage 4: AlphaZero (主力得分 agent) ---
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

# --- Stage 5: 汇总结果 + 生成 submission (figures, arena, submission.py) ---
log "START finalize"
"$PY" -m connectx.training.finalize_results \
  --results-dir results \
  --arena-games 100 \
  2>&1 | tee logs/finalize.log

log "ALL STAGES COMPLETE"
