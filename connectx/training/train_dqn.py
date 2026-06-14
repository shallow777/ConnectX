from __future__ import annotations

import argparse
import csv
from pathlib import Path

from connectx.agents.dqn import train_dqn_selfplay
from connectx.agents.reward_shaping import RewardShapingConfig
from connectx.envs.connectx_env import ConnectXConfig
from connectx.training.run_manifest import reward_shaping_fields, write_run_manifest


def train(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        run_dir,
        algorithm="dqn",
        episodes=args.episodes,
        device=args.device,
        **reward_shaping_fields(args.reward_shaping),
    )
    agent, curve = train_dqn_selfplay(
        args.episodes,
        config=ConnectXConfig(rows=args.rows, columns=args.columns, inarow=args.inarow),
        channels=args.channels,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        target_sync=args.target_sync,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        device=args.device,
        reward_shaping=RewardShapingConfig() if args.reward_shaping else None,
    )
    agent.save(run_dir / "dqn.pt")
    with (run_dir / "learning_curve.csv").open("w", newline="") as f:
        fieldnames = ["episode", "epsilon", "winner", "replay_size", "mean_loss"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve)
    write_run_manifest(
        run_dir,
        algorithm="dqn",
        episodes=args.episodes,
        final_mean_loss=float(curve[-1]["mean_loss"]) if curve else 0.0,
        status="completed",
        **reward_shaping_fields(args.reward_shaping),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standard-board DQN ConnectX baseline.")
    parser.add_argument("--run-dir", default="runs/dqn")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--inarow", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--target-sync", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--reward-shaping", action="store_true", help="Enable heuristic intermediate rewards.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
