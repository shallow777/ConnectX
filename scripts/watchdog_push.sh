#!/usr/bin/env bash
# Watchdog: monitor AlphaZero push training and auto-restart on failure/stall.
set -euo pipefail

WD=/root/autodl-tmp/ConnectX_new
LOG="$WD/logs/push_watchdog.log"
TRAIN_LOG="$WD/logs/alphazero_push.log"
PUSH_LOG="$WD/logs/train_alphazero_push.log"
SESSION=cx_az_push
INTERVAL="${1:-120}"
STALL_SEC="${2:-2400}"

cd "$WD"
mkdir -p logs

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

training_running() {
  pgrep -f '[p]ython -m connectx.training.train_alphazero.*alphazero_push' >/dev/null
}

training_complete() {
  grep -q '=== ALPHAZERO PUSH COMPLETE ===' "$PUSH_LOG" 2>/dev/null
}

last_gen_line() {
  grep '^\[alphazero\]' "$TRAIN_LOG" 2>/dev/null | tail -1 || true
}

log_age_sec() {
  if [[ -f "$TRAIN_LOG" ]]; then
    echo $(( $(date +%s) - $(stat -c %Y "$TRAIN_LOG") ))
  else
    echo 999999
  fi
}

ensure_tmux_training() {
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    log "START tmux session $SESSION"
    tmux new-session -d -s "$SESSION" "cd '$WD' && bash scripts/train_alphazero_push.sh"
    return 0
  fi
  if ! training_running; then
    if training_complete; then
      log "Training finished successfully; watchdog idle."
      return 2
    fi
    log "RESTART training in tmux $SESSION"
    tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
    sleep 2
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 1
    tmux new-session -d -s "$SESSION" "cd '$WD' && bash scripts/train_alphazero_push.sh"
    return 0
  fi
  return 1
}

check_errors() {
  if [[ ! -f "$TRAIN_LOG" ]]; then
    return
  fi
  local recent
  recent=$(tail -80 "$TRAIN_LOG" | grep -iE 'traceback|runtimeerror|cuda out of memory|killed process' || true)
  if [[ -n "$recent" ]]; then
    log "ERROR detected in alphazero_push.log — will restart on next cycle if process dead"
    echo "$recent" | tail -5 >> "$LOG"
  fi
}

log "=== watchdog started interval=${INTERVAL}s stall=${STALL_SEC}s ==="

while true; do
  {
    echo ""
    bash scripts/status_push.sh
    echo "--- recent ---"
    last_gen_line
  } >> "$LOG" 2>&1

  if training_complete && ! training_running; then
    log "DONE: push training complete."
    exit 0
  fi

  age=$(log_age_sec)
  if training_running && (( age > STALL_SEC )); then
    log "STALL: no log update for ${age}s — restarting training"
    pkill -f '[p]ython -m connectx.training.train_alphazero.*alphazero_push' 2>/dev/null || true
    sleep 3
    ensure_tmux_training || true
  elif ! training_running; then
    rc=0
    ensure_tmux_training || rc=$?
    if [[ "$rc" -eq 2 ]]; then
      exit 0
    fi
  fi

  check_errors
  sleep "$INTERVAL"
done
