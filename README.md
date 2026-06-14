# ConnectX RL Project

This repository implements the full course-project route:

- AlphaZero as the main scoring agent: residual policy-value network, PUCT MCTS, self-play, replay buffer, candidate gating.
- MaskablePPO self-play as the fallback and comparison agent.
- Tabular Q-learning on a small `4x5 connect-3` board.
- DQN on the standard `6x7 connect-4` board.
- Shared tactical safety layer: immediate win first, immediate opponent win block second.
- Single-file Kaggle submission generation with pure NumPy inference.

## Install on the server

```bash
pip install -e ".[all]"
```

## Stage 1 checks

```bash
python -m pytest tests
```

## PPO self-play

```bash
python -m connectx.training.train_ppo \
  --run-dir runs/ppo \
  --total-timesteps 500000 \
  --checkpoint-freq 50000 \
  --add-checkpoints-to-pool
```

The script writes checkpoints to `runs/ppo/checkpoints/` and negamax learning curve rows to `runs/ppo/negamax_curve.csv`.

## AlphaZero small end-to-end run

```bash
python -m connectx.training.train_alphazero \
  --run-dir runs/alphazero_small \
  --generations 3 \
  --selfplay-games 20 \
  --mcts-simulations 50 \
  --train-steps 100 \
  --arena-games 20 \
  --negamax-games 10
```

After confirming the curve improves, increase `--generations`, `--selfplay-games`, `--mcts-simulations`, `--train-steps`, and `--arena-games`.

## Baselines

```bash
python -m connectx.training.train_q_learning --run-dir runs/q_learning
python -m connectx.training.train_dqn --run-dir runs/dqn --device cuda
```

## Generate Kaggle submission

Choose automatically by local AlphaZero-vs-PPO arena:

```bash
python -m connectx.submission.make_submission \
  --kind auto \
  --alphazero-checkpoint runs/alphazero_small/checkpoints/alphazero_final.pt \
  --ppo-model runs/ppo/checkpoints/ppo_500000.zip \
  --output submission.py \
  --arena-games 200 \
  --validate
```

Or force AlphaZero:

```bash
python -m connectx.submission.make_submission \
  --kind alphazero \
  --alphazero-checkpoint runs/alphazero_small/checkpoints/alphazero_final.pt \
  --output submission.py \
  --validate
```

The generated `submission.py` defines `agent(obs, config)`, imports no `torch`, lazy-loads embedded NumPy weights, and wraps the learned policy with tactical safety.
