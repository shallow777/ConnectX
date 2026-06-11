"""MaskablePPO self-play 训练入口 (training entry point).

用 sb3-contrib 的 MaskablePPO 在 SelfPlayConnectXEnv 上做联盟式自我对弈
(league-style self-play):
- 学习方控制一方棋子, 对手从 OpponentPool 中随机采样;
- 每存一个 checkpoint, 可以用 --add-checkpoints-to-pool 把它冻结后加入对手池,
  这样后续训练要同时打赢历史版本, 避免遗忘 (avoid catastrophic forgetting);
- 每个 checkpoint 都会对战 Kaggle negamax, 学习曲线写到 negamax_curve.csv.

用法示例 (usage):
    python -m connectx.training.train_ppo --run-dir runs/ppo \
        --total-timesteps 500000 --checkpoint-freq 50000 --add-checkpoints-to-pool
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from connectx.agents.lookahead import wrap_with_tactical_safety
from connectx.agents.ppo_selfplay import OpponentPool, SelfPlayConnectXEnv, make_sb3_ppo_agent
from connectx.evaluation.kaggle_eval import evaluate_against_negamax


def build_env(opponent_pool: OpponentPool, args: argparse.Namespace) -> SelfPlayConnectXEnv:
    return SelfPlayConnectXEnv(
        opponent_pool=opponent_pool,
        rows=args.rows,
        columns=args.columns,
        inarow=args.inarow,
        randomize_player=True,   # 随机先后手, 让策略两边都会下
        tactical_safety=True,    # 训练时也套战术安全层, 减少低级失误样本
    )


def append_curve_row(path: Path, row: dict[str, Any]) -> None:
    """向学习曲线 CSV 追加一行; 文件不存在时先写表头."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(args: argparse.Namespace) -> None:
    # sb3 相关依赖只在训练时需要, 故延迟导入 (lazy import)
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

    opponent_pool = OpponentPool()
    env = Monitor(build_env(opponent_pool, args))
    eval_env = Monitor(build_env(opponent_pool, args))

    if args.resume:
        model = MaskablePPO.load(args.resume, env=env)
    else:
        model = MaskablePPO(
            "MultiInputPolicy",  # 观测是 dict(observation + action_mask)
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

    # 断点续训: 从 checkpoint 文件名 (ppo_<timesteps>.zip) 恢复已训练步数,
    # 并把更早的 checkpoint 重新加入对手池
    trained = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.stem.startswith("ppo_"):
            trained = int(resume_path.stem.split("_", 1)[1])
        if args.add_checkpoints_to_pool:
            for checkpoint_path in sorted(checkpoint_dir.glob("ppo_*.zip")):
                if int(checkpoint_path.stem.split("_", 1)[1]) <= trained:
                    opponent_pool.add_sb3_model(checkpoint_path)

    # 分块训练: 每 checkpoint_freq 步存一次档 + 评估一次 negamax
    while trained < args.total_timesteps:
        chunk = min(args.checkpoint_freq, args.total_timesteps - trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=trained == 0 and not args.resume, callback=eval_callback)
        trained += chunk

        checkpoint_path = checkpoint_dir / f"ppo_{trained}.zip"
        model.save(str(checkpoint_path))
        if args.add_checkpoints_to_pool:
            opponent_pool.add_sb3_model(checkpoint_path)

        # 外部评估: 套上战术安全层后对战 Kaggle negamax
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskablePPO self-play ConnectX agent.")
    parser.add_argument("--run-dir", default="runs/ppo")
    parser.add_argument("--resume", default=None, help="从已有 ppo_<steps>.zip 续训")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--inarow", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--negamax-games", type=int, default=20)
    parser.add_argument("--kaggle-timeout", type=float, default=2.0)
    # PPO 超参数 (hyperparameters)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--add-checkpoints-to-pool", action="store_true", help="把历史 checkpoint 加入对手池")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
