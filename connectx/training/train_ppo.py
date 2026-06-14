from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from connectx.agents.lookahead import wrap_with_tactical_safety
from connectx.agents.ppo_selfplay import OpponentPool, SelfPlayConnectXEnv, make_sb3_ppo_agent
from connectx.agents.reward_shaping import RewardShapingConfig
from connectx.agents.reward_shaping import RewardShapingConfig
from connectx.evaluation.kaggle_eval import evaluate_against_negamax
from connectx.training.run_manifest import reward_shaping_fields, write_run_manifest


def build_env(opponent_pool: OpponentPool, args: argparse.Namespace) -> SelfPlayConnectXEnv:
    shaping = RewardShapingConfig() if args.reward_shaping else None
    return SelfPlayConnectXEnv(
        opponent_pool=opponent_pool,
        rows=args.rows,
        columns=args.columns,
        inarow=args.inarow,
        randomize_player=True,
        tactical_safety=True,
        reward_shaping=shaping,
    )


def append_curve_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(args: argparse.Namespace) -> None:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError("Install stable-baselines3 and sb3-contrib on the training server.") from exc

    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curve_path = run_dir / "negamax_curve.csv"
    write_run_manifest(
        run_dir,
        algorithm="ppo",
        total_timesteps=args.total_timesteps,
        **reward_shaping_fields(args.reward_shaping),
    )

    opponent_pool = OpponentPool()
    env = Monitor(build_env(opponent_pool, args))
    eval_env = Monitor(build_env(opponent_pool, args))

    if args.resume:
        model = MaskablePPO.load(args.resume, env=env)
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=str(run_dir / "tensorboard"),
        )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )

    trained = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.stem.startswith("ppo_"):
            trained = int(resume_path.stem.split("_", 1)[1])
        if args.add_checkpoints_to_pool:
            for checkpoint_path in sorted(checkpoint_dir.glob("ppo_*.zip")):
                if int(checkpoint_path.stem.split("_", 1)[1]) <= trained:
                    opponent_pool.add_sb3_model(checkpoint_path)

    while trained < args.total_timesteps:
        chunk = min(args.checkpoint_freq, args.total_timesteps - trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=trained == 0 and not args.resume, callback=eval_callback)
        trained += chunk

        checkpoint_path = checkpoint_dir / f"ppo_{trained}.zip"
        model.save(str(checkpoint_path))
        if args.add_checkpoints_to_pool:
            opponent_pool.add_sb3_model(checkpoint_path)

        agent = wrap_with_tactical_safety(make_sb3_ppo_agent(checkpoint_path))
        result = evaluate_against_negamax(
            agent,
            games=args.negamax_games,
            rows=args.rows,
            columns=args.columns,
            inarow=args.inarow,
            timeout=args.kaggle_timeout,
        )
        append_curve_row(
            curve_path,
            {
                "timesteps": trained,
                "checkpoint": str(checkpoint_path),
                "negamax_games": result.games,
                "negamax_wins": result.wins,
                "negamax_losses": result.losses,
                "negamax_draws": result.draws,
                "negamax_win_rate": result.win_rate,
                "negamax_mean_reward": result.mean_reward,
            },
        )

    write_run_manifest(
        run_dir,
        algorithm="ppo",
        total_timesteps=trained,
        status="completed",
        **reward_shaping_fields(args.reward_shaping),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskablePPO self-play ConnectX agent.")
    parser.add_argument("--run-dir", default="runs/ppo")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--inarow", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--negamax-games", type=int, default=20)
    parser.add_argument("--kaggle-timeout", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--add-checkpoints-to-pool", action="store_true")
    parser.add_argument("--reward-shaping", action="store_true", help="Enable heuristic intermediate rewards.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Minimal end-to-end run: small timesteps, checkpoint, negamax eval, tensorboard.",
    )
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.total_timesteps = min(args.total_timesteps, 2048)
    args.checkpoint_freq = min(args.checkpoint_freq, 2048)
    args.eval_freq = min(args.eval_freq, 1024)
    args.negamax_games = min(args.negamax_games, 4)
    args.run_dir = str(Path(args.run_dir) / "smoke")


if __name__ == "__main__":
    parsed = parse_args()
    apply_smoke_defaults(parsed)
    train(parsed)
