#!/usr/bin/env bash
# Background monitor for AlphaZero push training.
set -euo pipefail
WD=/root/autodl-tmp/ConnectX_new
LOG="$WD/logs/push_monitor.log"
INTERVAL="${1:-300}"

cd "$WD"
echo "=== push monitor started $(date '+%F %T') interval=${INTERVAL}s ===" >> "$LOG"

while true; do
  {
    echo ""
    bash scripts/status_push.sh
    echo "--- recent ---"
    grep '^\[alphazero\]' logs/alphazero_push.log 2>/dev/null | tail -2 || true
  } >> "$LOG" 2>&1

  if ! pgrep -f '[p]ython -m connectx.training.train_alphazero.*alphazero_push' >/dev/null; then
    echo "=== TRAINING STOPPED $(date '+%F %T') ===" >> "$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
