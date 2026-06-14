#!/usr/bin/env bash
set -euo pipefail

WD=/root/autodl-tmp/ConnectX_new
cd "$WD"

echo "=== Shaped training status $(date '+%F %T') ==="

for algo in q_learning dqn ppo alphazero; do
  run="runs/${algo}_shaped"
  if [[ -f "$run/run_manifest.json" ]]; then
    status=$(python3 -c "import json; print(json.load(open('$run/run_manifest.json')).get('status','running'))" 2>/dev/null || echo "?")
    echo "$algo: $status ($run)"
  elif [[ -d "$run" ]] && [[ -n "$(ls -A "$run" 2>/dev/null)" ]]; then
    echo "$algo: running ($run)"
  else
    echo "$algo: pending ($run)"
  fi
done

if pgrep -f "[p]ython -m connectx.training.train_" >/dev/null; then
  echo "active: $(pgrep -af '[p]ython -m connectx.training.train_' | tr '\n' ' ')"
fi

if [[ -f results/shaped/training_journal.csv ]]; then
  echo "--- journal (last 5) ---"
  tail -5 results/shaped/training_journal.csv
fi

if [[ -f results/shaped/negamax_comparison.csv ]]; then
  echo "--- negamax results ---"
  column -t -s, results/shaped/negamax_comparison.csv 2>/dev/null || cat results/shaped/negamax_comparison.csv
fi
