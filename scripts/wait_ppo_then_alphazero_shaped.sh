#!/usr/bin/env bash
set -euo pipefail

WD=/root/autodl-tmp/ConnectX_new
cd "$WD"

echo "[$(date '+%F %T')] Waiting for PPO shaped to finish..."
while pgrep -f "[p]ython -m connectx.training.train_ppo.*ppo_shaped" >/dev/null; do
  sleep 120
done

echo "[$(date '+%F %T')] Starting AlphaZero shaped + export..."
bash scripts/train_alphazero_shaped.sh
