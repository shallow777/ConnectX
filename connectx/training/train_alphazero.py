"""AlphaZero 训练入口 (training entry point).

整体流程是经典的 AlphaZero 循环, 每一代 (generation) 做四件事:
1. Self-play: 用当前最强模型 (champion) 自我对弈, 产生带 MCTS 策略标签的样本;
2. Train: 从 replay buffer 采样, 训练一个候选模型 (candidate);
3. Gating: candidate 和 champion 打 arena, 胜率超过阈值才接受为新 champion;
4. Eval: 让当前 champion 对战 Kaggle 内置 negamax, 记录学习曲线.

用法示例 (usage):
    python -m connectx.training.train_alphazero --run-dir runs/alphazero \
        --device cuda --generations 30 --selfplay-games 40 --mcts-simulations 100
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Any

import torch

from connectx.agents.alphazero.agent import make_alphazero_agent_from_model
from connectx.agents.alphazero.mcts import MCTSConfig
from connectx.agents.alphazero.network import AlphaZeroNet, alphazero_loss, save_checkpoint
from connectx.agents.alphazero.replay_buffer import ReplayBuffer
from connectx.agents.alphazero.selfplay import SelfPlayConfig, run_self_play_parallel
from connectx.evaluation.arena import AgentSpec, evaluate_pair
from connectx.evaluation.kaggle_eval import evaluate_against_negamax


def append_curve_row(path: Path, row: dict[str, Any]) -> None:
    """向学习曲线 CSV 追加一行; 文件不存在时先写表头 (append one row, write header on first call)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_batches(
    model: AlphaZeroNet,
    buffer: ReplayBuffer,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    l2_weight: float,
    device: str,
) -> list[dict[str, float]]:
    """从 replay buffer 采样并做 `steps` 次梯度更新 (gradient steps on sampled batches)."""
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history: list[dict[str, float]] = []

    for _ in range(steps):
        batch = buffer.sample(batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)
        policies = torch.as_tensor(batch.policies, dtype=torch.float32, device=device)
        values = torch.as_tensor(batch.values, dtype=torch.float32, device=device)
        masks = torch.as_tensor(batch.masks, dtype=torch.bool, device=device)

        logits, pred_values = model(states)
        # 损失 = policy 交叉熵 + value MSE + L2 正则 (见 network.alphazero_loss)
        loss, metrics = alphazero_loss(logits, pred_values, policies, values, masks, model, l2_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history.append(metrics)

    model.eval()
    return history


def candidate_acceptance_rate(
    champion: AlphaZeroNet,
    candidate: AlphaZeroNet,
    *,
    games: int,
    simulations: int,
    device: str,
) -> float:
    """Gating 对局: 返回 candidate 对 champion 的胜率 (win rate of candidate vs champion)."""
    champion_agent = AgentSpec(
        "champion",
        make_alphazero_agent_from_model(champion, simulations=simulations, device=device, tactical_safety=True),
    )
    candidate_agent = AgentSpec(
        "candidate",
        make_alphazero_agent_from_model(candidate, simulations=simulations, device=device, tactical_safety=True),
    )
    stats = evaluate_pair(candidate_agent, champion_agent, games=games)
    return stats.win_rate("candidate")


def train(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curve_path = run_dir / "negamax_curve.csv"
    buffer_path = run_dir / "replay_buffer.npz"

    # 支持断点续训: --resume-checkpoint 加载旧模型, --resume-buffer 加载旧样本池
    if args.resume_checkpoint:
        payload = torch.load(args.resume_checkpoint, map_location=args.device)
        network_config = dict(payload["network_config"])
        network_config.pop("l2_weight", None)
        model = AlphaZeroNet(**network_config)
        model.load_state_dict(payload["model_state_dict"])
    else:
        model = AlphaZeroNet(
            rows=args.rows,
            columns=args.columns,
            channels=args.channels,
            residual_blocks=args.residual_blocks,
        )
    model.to(args.device)
    model.eval()

    buffer = ReplayBuffer.load(buffer_path, capacity=args.replay_capacity) if args.resume_buffer and buffer_path.exists() else ReplayBuffer(args.replay_capacity)

    for generation in range(args.generations):
        # 每代先落盘当前 champion, self-play worker 会从该 checkpoint 加载模型
        champion_path = checkpoint_dir / f"generation_{generation:04d}_champion.pt"
        save_checkpoint(champion_path, model, extra={"generation": generation})

        # --- Step 1: self-play 收集训练样本 ---
        selfplay_config = SelfPlayConfig(
            games=args.selfplay_games,
            max_moves=args.rows * args.columns,
            temperature_moves=args.temperature_moves,
            rows=args.rows,
            columns=args.columns,
            inarow=args.inarow,
            device=args.device,
        )
        mcts_config = MCTSConfig(
            simulations=args.mcts_simulations,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
        )
        samples = run_self_play_parallel(champion_path, selfplay_config, mcts_config, workers=args.workers)
        buffer.add_game(samples)
        buffer.save(buffer_path)

        # 样本太少没法采样一个 batch, 跳过本代训练
        if len(buffer) < args.batch_size:
            append_curve_row(
                curve_path,
                {
                    "generation": generation,
                    "buffer_size": len(buffer),
                    "accepted": False,
                    "candidate_win_rate": 0.0,
                    "negamax_win_rate": 0.0,
                    "note": "buffer_smaller_than_batch",
                },
            )
            continue

        # --- Step 2: 复制 champion 训练出 candidate ---
        candidate = copy.deepcopy(model)
        metrics = train_batches(
            candidate,
            buffer,
            steps=args.train_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2_weight=args.l2_weight,
            device=args.device,
        )

        # --- Step 3: gating, candidate 胜率超过阈值才接受 (AlphaZero-style gating) ---
        win_rate = candidate_acceptance_rate(
            model,
            candidate,
            games=args.arena_games,
            simulations=args.eval_mcts_simulations,
            device=args.device,
        )
        accepted = win_rate > args.accept_threshold
        if accepted:
            model = candidate

        accepted_path = checkpoint_dir / f"generation_{generation:04d}_{'accepted' if accepted else 'rejected'}.pt"
        save_checkpoint(
            accepted_path,
            candidate,
            extra={
                "generation": generation,
                "candidate_win_rate": win_rate,
                "accepted": accepted,
                "last_train_metrics": metrics[-1] if metrics else {},
            },
        )

        # --- Step 4: 对战 Kaggle 内置 negamax, 记录外部强度曲线 ---
        negamax_agent = make_alphazero_agent_from_model(
            model,
            simulations=args.eval_mcts_simulations,
            device=args.device,
            tactical_safety=True,
        )
        negamax_result = evaluate_against_negamax(
            negamax_agent,
            games=args.negamax_games,
            rows=args.rows,
            columns=args.columns,
            inarow=args.inarow,
            timeout=args.kaggle_timeout,
        )
        append_curve_row(
            curve_path,
            {
                "generation": generation,
                "buffer_size": len(buffer),
                "accepted": accepted,
                "candidate_win_rate": win_rate,
                "negamax_games": negamax_result.games,
                "negamax_wins": negamax_result.wins,
                "negamax_losses": negamax_result.losses,
                "negamax_draws": negamax_result.draws,
                "negamax_win_rate": negamax_result.win_rate,
                "negamax_mean_reward": negamax_result.mean_reward,
                "note": "",
            },
        )
        print(
            f"[alphazero] gen {generation + 1}/{args.generations} accepted={accepted} "
            f"candidate_wr={win_rate:.3f} negamax_wr={negamax_result.win_rate:.3f} buffer={len(buffer)}",
            flush=True,
        )

    final_path = checkpoint_dir / "alphazero_final.pt"
    save_checkpoint(final_path, model, extra={"generation": args.generations})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AlphaZero ConnectX with self-play.")
    parser.add_argument("--run-dir", default="runs/alphazero")
    parser.add_argument("--resume-checkpoint", default=None, help="从已有 checkpoint 续训 (warm start)")
    parser.add_argument("--resume-buffer", action="store_true", help="同时恢复 replay buffer")
    parser.add_argument("--device", default="cpu")
    # 棋盘配置 (board config)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--inarow", type=int, default=4)
    # 网络结构 (network)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=3)
    # self-play / MCTS
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--selfplay-games", type=int, default=20, help="每代自我对弈局数")
    parser.add_argument("--workers", type=int, default=1, help="self-play 并行进程数")
    parser.add_argument("--mcts-simulations", type=int, default=50, help="self-play 时每步 MCTS 模拟数")
    parser.add_argument("--eval-mcts-simulations", type=int, default=50, help="评估时每步 MCTS 模拟数")
    parser.add_argument("--temperature-moves", type=int, default=8, help="前 N 步用温度采样增加开局多样性")
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    # 训练 (optimization)
    parser.add_argument("--replay-capacity", type=int, default=500_000)
    parser.add_argument("--train-steps", type=int, default=200, help="每代梯度更新步数")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    # gating / 评估
    parser.add_argument("--arena-games", type=int, default=100, help="candidate vs champion 对局数")
    parser.add_argument("--accept-threshold", type=float, default=0.55, help="candidate 接受阈值胜率")
    parser.add_argument("--negamax-games", type=int, default=20)
    parser.add_argument("--kaggle-timeout", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
