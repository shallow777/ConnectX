#!/usr/bin/env bash
# Lightweight watchdog: restart sequential training if cx_train dies.
set -uo pipefail

WD=/root/autodl-tmp/ConnectX_new
SCRIPT="$WD/scripts/train_sequential.sh"
LOG="$WD/logs/watchdog.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

while true; do
  if pgrep -f "ConnectX_new.*connectx.training.train_" >/dev/null 2>&1; then
    sleep 120
    continue
  fi
  if ! tmux has-session -t cx_train 2>/dev/null; then
    if [[ -f "$WD/logs/overnight_complete.marker" ]]; then
      log "All stages including overnight complete; watchdog exiting."
      break
    fi
    if [[ -f "$WD/runs/alphazero/checkpoints/alphazero_final.pt" ]] \
       && grep -q "ALL STAGES COMPLETE" "$WD/logs/train_all.log" 2>/dev/null \
       && [[ ! -f "$WD/logs/overnight_complete.marker" ]]; then
      log "Main done, overnight pending; restarting train_sequential.sh"
    fi
    log "cx_train missing; restarting train_sequential.sh"
    tmux new-session -d -s cx_train "bash $SCRIPT"
  fi
  sleep 120
done
