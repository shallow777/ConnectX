#!/usr/bin/env bash
# Quick status for AlphaZero push training.
set -euo pipefail
WD=/root/autodl-tmp/ConnectX_new
cd "$WD"

echo "=== AlphaZero Push $(date '+%F %T') ==="

if pgrep -f "[p]ython -m connectx.training.train_alphazero.*alphazero_push" >/dev/null; then
  echo "status: RUNNING"
  workers=$(pgrep -af "[p]ython -m connectx.training.train_alphazero.*alphazero_push" | head -1 | sed -n 's/.*--workers \([0-9]*\).*/\1/p')
  [[ -n "$workers" ]] && echo "selfplay_workers: $workers (MCTS on CPU, batched GPU inference server)"
else
  echo "status: STOPPED"
fi

if [[ -f runs/alphazero_push/negamax_curve.csv ]]; then
  gens=$(($(wc -l < runs/alphazero_push/negamax_curve.csv) - 1))
  echo "generations logged: $gens / 80"
  echo "--- last 3 gens ---"
  tail -3 runs/alphazero_push/negamax_curve.csv | column -t -s, 2>/dev/null || tail -3 runs/alphazero_push/negamax_curve.csv
  best=$(/root/miniconda3/envs/ConnectX/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path("runs/alphazero_push/negamax_curve.csv")
if p.exists():
    df = pd.read_csv(p)
    if not df.empty and "negamax_win_rate" in df.columns:
        row = df.loc[df["negamax_win_rate"].idxmax()]
        print(f"best gen {int(row['generation'])} negamax={row['negamax_win_rate']:.1%} accepted={row.get('accepted','')}")
PY
)
  [[ -n "$best" ]] && echo "peak: $best"
else
  echo "generations logged: 0 (self-play in progress...)"
fi

if [[ -f runs/alphazero_push/replay_buffer.npz ]]; then
  size=$(du -h runs/alphazero_push/replay_buffer.npz | cut -f1)
  echo "replay buffer: $size"
fi

nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'

echo "early_stop: negamax>=97% (60-game eval) OR 18 gens plateau (after min 20 gens, best>=93%)"
echo "note: local negamax != Kaggle score; keep training until strict criteria met"
if [[ -f runs/alphazero_push/best_checkpoint.txt ]]; then
  echo "--- best so far ---"
  cat runs/alphazero_push/best_checkpoint.txt
fi

echo "logs: tail -f logs/alphazero_push.log"
echo "tmux: attach -t cx_az_push"
