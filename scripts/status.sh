#!/usr/bin/env bash
# Print one-line training status snapshot (safe to run repeatedly).
set -uo pipefail

WD=/root/autodl-tmp/ConnectX_new
cd "$WD"

stage() {
  if pgrep -f "connectx.training.train_ppo" >/dev/null; then echo -n "PPO(running) "
  elif [[ -f runs/ppo/checkpoints/ppo_500000.zip ]]; then echo -n "PPO(done) "
  else echo -n "PPO(pending) "; fi

  if pgrep -f "connectx.training.train_dqn" >/dev/null; then echo -n "DQN(running) "
  elif [[ -f runs/dqn/dqn.pt ]]; then echo -n "DQN(done) "
  else echo -n "DQN(pending) "; fi

  if pgrep -f "connectx.training.train_alphazero" >/dev/null; then echo -n "AZ(running) "
  elif [[ -f runs/alphazero/checkpoints/alphazero_final.pt ]] && [[ $(wc -l < runs/alphazero/negamax_curve.csv 2>/dev/null || echo 0) -ge 31 ]]; then echo -n "AZ(done) "
  else echo -n "AZ(pending) "; fi
}

echo "[$(date '+%F %T')] $(stage)"
echo "tmux: $(tmux ls 2>/dev/null | tr '\n' ' ')"
if [[ -f runs/ppo/negamax_curve.csv ]]; then
  echo "PPO negamax last: $(tail -1 runs/ppo/negamax_curve.csv)"
fi
if [[ -f runs/alphazero/negamax_curve.csv ]]; then
  echo "AZ negamax last: $(tail -1 runs/alphazero/negamax_curve.csv)"
fi
tail -3 logs/dqn_train.log 2>/dev/null | sed 's/^/dqn: /'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
