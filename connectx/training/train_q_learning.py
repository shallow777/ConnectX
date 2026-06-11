from __future__ import annotations

import argparse
import csv
from pathlib import Path

from connectx.agents.q_learning import train_tabular_q_learning
from connectx.envs.connectx_env import ConnectXConfig


def train(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    agent, curve = train_tabular_q_learning(
        args.episodes,
        config=ConnectXConfig(rows=args.rows, columns=args.columns, inarow=args.inarow),
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
    )
    agent.save(run_dir / "q_learning.pkl")
    with (run_dir / "learning_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "epsilon", "winner", "q_states"])
        writer.writeheader()
        writer.writerows(curve)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train small-board tabular Q-learning ConnectX baseline.")
    parser.add_argument("--run-dir", default="runs/q_learning")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--inarow", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
