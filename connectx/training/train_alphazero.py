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
from connectx.agents.reward_shaping import RewardShapingConfig
from connectx.evaluation.arena import AgentSpec, evaluate_pair
from connectx.evaluation.kaggle_eval import evaluate_against_negamax
from connectx.training.run_manifest import reward_shaping_fields, write_run_manifest


def append_curve_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def learning_rate_for_generation(
    generation: int,
    *,
    base_lr: float,
    final_lr: float,
    total_generations: int,
    decay_start_generation: int,
) -> float:
    if final_lr <= 0 or final_lr >= base_lr or generation < decay_start_generation:
        return base_lr
    span = max(1, total_generations - decay_start_generation)
    progress = min(1.0, (generation - decay_start_generation) / span)
    return base_lr + (final_lr - base_lr) * progress


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


def checkpoint_for_generation(checkpoint_dir: Path, generation: int, accepted: bool) -> Path:
    accepted_path = checkpoint_dir / f"generation_{generation:04d}_accepted.pt"
    if accepted and accepted_path.exists():
        return accepted_path
    champion_path = checkpoint_dir / f"generation_{generation:04d}_champion.pt"
    if champion_path.exists():
        return champion_path
    rejected_path = checkpoint_dir / f"generation_{generation:04d}_rejected.pt"
    return rejected_path


def best_checkpoint_from_run(run_dir: Path) -> Path | None:
    import pandas as pd

    curve_path = run_dir / "negamax_curve.csv"
    checkpoint_dir = run_dir / "checkpoints"
    if not curve_path.exists() or not checkpoint_dir.exists():
        final = checkpoint_dir / "alphazero_final.pt"
        return final if final.exists() else None
    frame = pd.read_csv(curve_path)
    if frame.empty or "negamax_win_rate" not in frame.columns:
        final = checkpoint_dir / "alphazero_final.pt"
        return final if final.exists() else None
    row = frame.loc[frame["negamax_win_rate"].idxmax()]
    generation = int(row["generation"])
    accepted = bool(row.get("accepted", False))
    path = checkpoint_for_generation(checkpoint_dir, generation, accepted)
    return path if path.exists() else checkpoint_dir / "alphazero_final.pt"


def should_early_stop(
    *,
    generation: int,
    negamax_win_rate: float,
    best_negamax: float,
    generations_without_improvement: int,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if generation + 1 < args.early_stop_min_generations:
        return False, ""
    if args.early_stop_negamax > 0 and negamax_win_rate >= args.early_stop_negamax:
        return True, f"negamax_win_rate {negamax_win_rate:.3f} >= {args.early_stop_negamax:.3f}"
    if (
        args.early_stop_patience > 0
        and generations_without_improvement >= args.early_stop_patience
        and best_negamax >= args.early_stop_min_negamax
    ):
        return (
            True,
            f"no negamax improvement for {generations_without_improvement} gens "
            f"(best={best_negamax:.3f})",
        )
    return False, ""


def train(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curve_path = run_dir / "negamax_curve.csv"
    buffer_path = run_dir / "replay_buffer.npz"

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

    replay_max_generations = args.replay_max_generations or None
    buffer = (
        ReplayBuffer.load(
            buffer_path,
            capacity=args.replay_capacity,
            max_generations=replay_max_generations,
        )
        if args.resume_buffer and buffer_path.exists()
        else ReplayBuffer(args.replay_capacity, max_generations=replay_max_generations)
    )
    write_run_manifest(
        run_dir,
        algorithm="alphazero",
        generations=args.generations,
        device=args.device,
        **reward_shaping_fields(args.reward_shaping),
    )

    best_negamax = -1.0
    best_generation = -1
    generations_without_improvement = 0
    early_stop_reason = ""
    if curve_path.exists():
        import pandas as pd

        frame = pd.read_csv(curve_path)
        if not frame.empty and "negamax_win_rate" in frame.columns:
            peak_idx = frame["negamax_win_rate"].idxmax()
            row = frame.loc[peak_idx]
            best_negamax = float(row["negamax_win_rate"])
            best_generation = int(row["generation"])
            generations_without_improvement = max(0, len(frame) - 1 - int(peak_idx))
            best_ckpt = checkpoint_for_generation(
                checkpoint_dir,
                best_generation,
                bool(row.get("accepted", False)),
            )
            if best_ckpt.exists():
                (run_dir / "best_checkpoint.txt").write_text(
                    f"generation={best_generation}\n"
                    f"negamax_win_rate={best_negamax}\n"
                    f"checkpoint={best_ckpt}\n"
                )

    for generation in range(args.generations):
        champion_path = checkpoint_dir / f"generation_{generation:04d}_champion.pt"
        save_checkpoint(champion_path, model, extra={"generation": generation})

        selfplay_config = SelfPlayConfig(
            games=args.selfplay_games,
            max_moves=args.rows * args.columns,
            temperature_moves=args.temperature_moves,
            rows=args.rows,
            columns=args.columns,
            inarow=args.inarow,
            device=args.device,
            reward_shaping=RewardShapingConfig(gamma=args.gamma) if args.reward_shaping else None,
            gamma=args.gamma,
            mcts_simulations_high=args.mcts_simulations_high,
            high_quality_prob=args.high_quality_prob,
            mirror_augment=not args.no_mirror_augment,
        )
        mcts_config = MCTSConfig(
            simulations=args.mcts_simulations,
            eval_batch_size=args.mcts_eval_batch_size,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
        )
        samples = run_self_play_parallel(
            champion_path,
            selfplay_config,
            mcts_config,
            workers=args.workers,
            inference_device=args.device,
            inference_batch_size=args.inference_batch_size,
            inference_max_wait_ms=args.inference_max_wait_ms,
        )
        buffer.add_game(samples, generation=generation)
        buffer.save(buffer_path)

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

        candidate = copy.deepcopy(model)
        generation_lr = learning_rate_for_generation(
            generation,
            base_lr=args.learning_rate,
            final_lr=args.learning_rate_final,
            total_generations=args.generations,
            decay_start_generation=args.lr_decay_start,
        )
        metrics = train_batches(
            candidate,
            buffer,
            steps=args.train_steps,
            batch_size=args.batch_size,
            learning_rate=generation_lr,
            l2_weight=args.l2_weight,
            device=args.device,
        )
        if args.no_gating:
            win_rate = 1.0
            accepted = True
        else:
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
            f"candidate_wr={win_rate:.3f} negamax_wr={negamax_result.win_rate:.3f} "
            f"lr={generation_lr:.2e} buffer={len(buffer)}",
            flush=True,
        )

        if negamax_result.win_rate > best_negamax + 1e-6:
            best_negamax = negamax_result.win_rate
            best_generation = generation
            generations_without_improvement = 0
            best_ckpt = checkpoint_for_generation(checkpoint_dir, generation, accepted)
            save_checkpoint(
                checkpoint_dir / "alphazero_best.pt",
                model,
                extra={
                    "generation": generation,
                    "negamax_win_rate": negamax_result.win_rate,
                    "source_checkpoint": str(best_ckpt),
                },
            )
            (run_dir / "best_checkpoint.txt").write_text(
                f"generation={generation}\n"
                f"negamax_win_rate={negamax_result.win_rate}\n"
                f"checkpoint={best_ckpt}\n"
            )
        else:
            generations_without_improvement += 1

        stop, reason = should_early_stop(
            generation=generation,
            negamax_win_rate=negamax_result.win_rate,
            best_negamax=best_negamax,
            generations_without_improvement=generations_without_improvement,
            args=args,
        )
        if stop:
            early_stop_reason = reason
            print(f"[alphazero] early stop at gen {generation + 1}: {reason}", flush=True)
            break

    final_path = checkpoint_dir / "alphazero_final.pt"
    save_checkpoint(final_path, model, extra={"generation": generation, "early_stop": bool(early_stop_reason)})
    write_run_manifest(
        run_dir,
        algorithm="alphazero",
        generations=generation + 1,
        status="early_stopped" if early_stop_reason else "completed",
        early_stop_reason=early_stop_reason,
        best_negamax_win_rate=best_negamax,
        best_generation=best_generation,
        final_checkpoint=str(final_path),
        best_checkpoint=str(checkpoint_dir / "alphazero_best.pt") if best_generation >= 0 else "",
        **reward_shaping_fields(args.reward_shaping),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AlphaZero ConnectX with self-play.")
    parser.add_argument("--run-dir", default="runs/alphazero")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--resume-buffer", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--inarow", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--selfplay-games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=64,
        help="GPU batch size for parallel self-play inference server.",
    )
    parser.add_argument(
        "--inference-max-wait-ms",
        type=float,
        default=2.0,
        help="Max wait when batching GPU inference requests from self-play workers.",
    )
    parser.add_argument("--mcts-simulations", type=int, default=50)
    parser.add_argument(
        "--mcts-simulations-high",
        type=int,
        default=0,
        help="High-quality MCTS sims for a random subset of self-play moves (0 disables).",
    )
    parser.add_argument(
        "--high-quality-prob",
        type=float,
        default=0.25,
        help="Fraction of self-play moves that use high sims and enter the replay buffer.",
    )
    parser.add_argument(
        "--no-mirror-augment",
        action="store_true",
        help="Disable horizontal mirror augmentation in self-play.",
    )
    parser.add_argument("--mcts-eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-mcts-simulations", type=int, default=50)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--replay-capacity", type=int, default=500_000)
    parser.add_argument(
        "--replay-max-generations",
        type=int,
        default=10,
        help="Keep replay samples from only the most recent N generations (0 disables).",
    )
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--learning-rate-final",
        type=float,
        default=2e-4,
        help="Linear LR decay target in later generations (0 disables decay).",
    )
    parser.add_argument(
        "--lr-decay-start",
        type=int,
        default=20,
        help="Generation index where LR decay begins.",
    )
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--arena-games", type=int, default=150)
    parser.add_argument("--accept-threshold", type=float, default=0.55)
    parser.add_argument(
        "--no-gating",
        action="store_true",
        help="Always accept the latest candidate (skip arena gating).",
    )
    parser.add_argument("--negamax-games", type=int, default=20)
    parser.add_argument("--kaggle-timeout", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward-shaping", action="store_true", help="Add heuristic shaping to value targets.")
    parser.add_argument(
        "--early-stop-negamax",
        type=float,
        default=0.0,
        help="Stop when negamax win rate reaches this threshold (0 disables).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many generations without negamax improvement (0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-generations",
        type=int,
        default=5,
        help="Minimum generations before early stopping can trigger.",
    )
    parser.add_argument(
        "--early-stop-min-negamax",
        type=float,
        default=0.85,
        help="Patience-based early stop requires at least this best negamax win rate.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Minimal end-to-end run: 1 generation, tiny self-play, checkpoint, negamax eval.",
    )
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.generations = min(args.generations, 1)
    args.selfplay_games = min(args.selfplay_games, 8)
    args.mcts_simulations = min(args.mcts_simulations, 16)
    args.eval_mcts_simulations = min(args.eval_mcts_simulations, 16)
    args.train_steps = min(args.train_steps, 8)
    args.batch_size = min(args.batch_size, 32)
    args.arena_games = min(args.arena_games, 4)
    args.negamax_games = min(args.negamax_games, 4)
    args.mcts_simulations_high = 0
    args.workers = 1
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.run_dir = str(Path(args.run_dir) / "smoke")


if __name__ == "__main__":
    parsed = parse_args()
    apply_smoke_defaults(parsed)
    train(parsed)
