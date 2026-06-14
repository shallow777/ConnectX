#!/usr/bin/env bash
# Wait only for DQN shaped (short); skip waiting for alphazero_shaped (different hypothesis).
set -euo pipefail

WD=/root/autodl-tmp/ConnectX_new
cd "$WD"

echo "[$(date '+%F %T')] Waiting for DQN shaped to free GPU..."
while pgrep -f "[p]ython -m connectx.training.train_dqn.*dqn_shaped" >/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] Starting AlphaZero push (data + search)..."
bash scripts/train_alphazero_push.sh
