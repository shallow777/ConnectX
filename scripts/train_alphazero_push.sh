#!/usr/bin/env bash
# Phase-2 AlphaZero: scale self-play DATA + MCTS DEPTH (same 64ch/3-block net as overnight).
set -euo pipefail

PY=/root/miniconda3/envs/ConnectX/bin/python
WD=/root/autodl-tmp/ConnectX_new
cd "$WD"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p logs runs/alphazero_push/checkpoints

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/train_alphazero_push.log; }

BEST_CKPT="runs/alphazero_overnight/checkpoints/generation_0038_champion.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  BEST_CKPT=$(ls -1 runs/alphazero_overnight/checkpoints/generation_*_accepted.pt 2>/dev/null | sort -V | tail -1 || true)
fi
if [[ -z "$BEST_CKPT" || ! -f "$BEST_CKPT" ]]; then
  BEST_CKPT="runs/alphazero_overnight/checkpoints/alphazero_final.pt"
fi

BUFFER_ARGS=()
RESUME_ARGS=(--resume-checkpoint "$BEST_CKPT")

if [[ -f runs/alphazero_push/checkpoints/alphazero_best.pt ]] || [[ -f runs/alphazero_push/best_checkpoint.txt ]]; then
  PUSH_BEST=$($PY - <<'PY'
from connectx.training.train_alphazero import best_checkpoint_from_run
from pathlib import Path
p = best_checkpoint_from_run(Path("runs/alphazero_push"))
print(p or "")
PY
)
  if [[ -n "$PUSH_BEST" && -f "$PUSH_BEST" ]]; then
    RESUME_ARGS=(--resume-checkpoint "$PUSH_BEST")
    BUFFER_ARGS=(--resume-buffer)
    log "Resume push from best checkpoint: $PUSH_BEST"
  fi
elif [[ -f runs/alphazero_push/replay_buffer.npz ]]; then
  BUFFER_ARGS=(--resume-buffer)
elif [[ -f runs/alphazero_overnight/replay_buffer.npz ]]; then
  cp -f runs/alphazero_overnight/replay_buffer.npz runs/alphazero_push/replay_buffer.npz
  BUFFER_ARGS=(--resume-buffer)
fi

SELFPLAY_WORKERS=8
INFERENCE_BATCH_SIZE=128
MCTS_EVAL_BATCH_SIZE=16

log "=== ALPHAZERO PUSH phase-3 (mirror aug, playout-cap, LR decay, no-gating, workers=${SELFPLAY_WORKERS}) seed=${RESUME_ARGS[*]} ==="

# Optimizations: mirror data aug, 400/800 playout-cap randomization, LR decay, recent-gen replay, no gating.
# Parallel self-play: N CPU workers + 1 shared GPU batched inference server.
"$PY" -m connectx.training.train_alphazero \
  --run-dir runs/alphazero_push \
  --device cuda \
  "${RESUME_ARGS[@]}" \
  "${BUFFER_ARGS[@]}" \
  --generations 100 \
  --selfplay-games 120 \
  --workers "$SELFPLAY_WORKERS" \
  --inference-batch-size "$INFERENCE_BATCH_SIZE" \
  --inference-max-wait-ms 1 \
  --mcts-eval-batch-size "$MCTS_EVAL_BATCH_SIZE" \
  --mcts-simulations 400 \
  --mcts-simulations-high 800 \
  --high-quality-prob 0.25 \
  --eval-mcts-simulations 300 \
  --train-steps 1000 \
  --batch-size 256 \
  --replay-capacity 800000 \
  --replay-max-generations 10 \
  --learning-rate 1e-3 \
  --learning-rate-final 2e-4 \
  --lr-decay-start 20 \
  --no-gating \
  --negamax-games 60 \
  --early-stop-negamax 0.97 \
  --early-stop-patience 18 \
  --early-stop-min-generations 20 \
  --early-stop-min-negamax 0.93 \
  2>&1 | tee -a logs/alphazero_push.log

EXPORT_CKPT=$($PY - <<'PY'
from connectx.training.train_alphazero import best_checkpoint_from_run
from pathlib import Path
run = Path("runs/alphazero_push")
best = best_checkpoint_from_run(run)
if best:
    print(best)
else:
    final = run / "checkpoints/alphazero_final.pt"
    print(final if final.exists() else "")
PY
)

log "Export submission from best checkpoint: $EXPORT_CKPT"
"$PY" -m connectx.submission.make_all_submissions \
  --output-dir submission \
  --alphazero-checkpoint "$EXPORT_CKPT" \
  --tag push \
  --notes "Push peak negamax checkpoint; Kaggle score may differ from local negamax" \
  --validate \
  2>&1 | tee logs/export_push.log

log "Finalize"
"$PY" -m connectx.training.finalize_results \
  --results-dir results \
  --alphazero-checkpoint "$EXPORT_CKPT" \
  --alphazero-simulations 200 \
  --negamax-games 40 \
  2>&1 | tee logs/finalize_push.log

log "=== ALPHAZERO PUSH COMPLETE ==="
