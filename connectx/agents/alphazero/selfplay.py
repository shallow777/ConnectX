from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from connectx.agents.alphazero.inference_server import RemoteEvaluator, start_inference_server
from connectx.agents.alphazero.mcts import AlphaZeroMCTS, MCTSConfig, TorchEvaluator
from connectx.agents.alphazero.network import AlphaZeroNet, load_checkpoint
from connectx.agents.reward_shaping import (
    RewardShapingConfig,
    compute_alphazero_value_targets,
    compute_shaping_delta,
)
from connectx.envs.connectx_env import (
    ConnectXConfig,
    check_winner,
    encode_board,
    is_draw,
    mirror_encoded_state,
    mirror_mask,
    mirror_policy,
    next_board,
    opponent_mark,
    valid_action_mask,
)


SelfPlaySample = tuple[np.ndarray, np.ndarray, float, np.ndarray]


@dataclass(frozen=True)
class SelfPlayConfig:
    games: int = 20
    max_moves: int = 42
    temperature_moves: int = 8
    rows: int = 6
    columns: int = 7
    inarow: int = 4
    device: str = "cpu"
    reward_shaping: RewardShapingConfig | None = None
    gamma: float = 0.99
    mcts_simulations_high: int = 0
    high_quality_prob: float = 0.25
    mirror_augment: bool = True

    @property
    def game_config(self) -> ConnectXConfig:
        return ConnectXConfig(rows=self.rows, columns=self.columns, inarow=self.inarow)


def play_self_play_game(
    selfplay_config: SelfPlayConfig,
    mcts_config: MCTSConfig,
    *,
    model: AlphaZeroNet | None = None,
    evaluator: TorchEvaluator | RemoteEvaluator | None = None,
) -> list[SelfPlaySample]:
    config = selfplay_config.game_config
    if evaluator is None:
        if model is None:
            raise ValueError("play_self_play_game requires model or evaluator")
        evaluator = TorchEvaluator(model, config, device=selfplay_config.device)
    mcts = AlphaZeroMCTS(evaluator, config, mcts_config)

    board = [0] * (config.rows * config.columns)
    mark = 1
    history: list[tuple[np.ndarray, np.ndarray, int, np.ndarray]] = []
    train_ply_indices: list[int] = []
    marks: list[int] = []
    shaping_rewards: list[float] = []
    winner = 0

    use_high_sims = (
        selfplay_config.mcts_simulations_high > 0
        and selfplay_config.mcts_simulations_high > mcts_config.simulations
    )

    for move_idx in range(selfplay_config.max_moves):
        temperature = 1.0 if move_idx < selfplay_config.temperature_moves else 1e-6
        mask = valid_action_mask(board, config.rows, config.columns)
        high_quality = use_high_sims and np.random.random() < selfplay_config.high_quality_prob
        sims = selfplay_config.mcts_simulations_high if high_quality else mcts_config.simulations
        policy = mcts.search(
            board,
            mark,
            add_noise=True,
            temperature=temperature,
            simulations=sims,
        )
        if policy.sum() <= 0:
            legal = np.flatnonzero(mask)
            action = int(np.random.choice(legal))
        else:
            action = int(np.random.choice(np.arange(config.columns), p=policy / policy.sum()))

        before = list(board)
        ply_index = len(marks)
        if not use_high_sims or high_quality:
            encoded = encode_board(board, mark, config.rows, config.columns)
            history.append((encoded, policy.astype(np.float32), mark, mask.astype(bool)))
            train_ply_indices.append(ply_index)
        board = next_board(board, action, mark, config.rows, config.columns)
        marks.append(mark)
        shaping_rewards.append(
            compute_shaping_delta(before, board, mark, config, selfplay_config.reward_shaping)
        )

        if check_winner(board, mark, config.rows, config.columns, config.inarow):
            winner = mark
            break
        if is_draw(board, config.rows, config.columns):
            winner = 0
            break
        mark = opponent_mark(mark)

    gamma = (
        selfplay_config.reward_shaping.gamma
        if selfplay_config.reward_shaping is not None
        else selfplay_config.gamma
    )
    value_targets = compute_alphazero_value_targets(
        marks,
        winner,
        shaping_rewards,
        gamma=gamma,
    )

    samples: list[SelfPlaySample] = []
    columns = config.columns
    for (encoded, policy, _sample_mark, mask), ply_index in zip(history, train_ply_indices):
        value = value_targets[ply_index]
        samples.append((encoded, policy, float(value), mask))
        if selfplay_config.mirror_augment:
            samples.append(
                (
                    mirror_encoded_state(encoded, columns),
                    mirror_policy(policy, columns),
                    float(value),
                    mirror_mask(mask, columns),
                )
            )
    return samples


def run_self_play(
    model: AlphaZeroNet,
    selfplay_config: SelfPlayConfig,
    mcts_config: MCTSConfig,
) -> list[SelfPlaySample]:
    samples: list[SelfPlaySample] = []
    for _ in range(selfplay_config.games):
        samples.extend(play_self_play_game(selfplay_config, mcts_config, model=model))
    return samples


def _worker_local_model(args: tuple[str, dict[str, Any], dict[str, Any], int]) -> list[SelfPlaySample]:
    checkpoint_path, selfplay_kwargs, mcts_kwargs, games = args
    model, _payload = load_checkpoint(checkpoint_path, map_location=selfplay_kwargs.get("device", "cpu"))
    config = SelfPlayConfig(**selfplay_kwargs)
    mcts = MCTSConfig(**mcts_kwargs)
    samples: list[SelfPlaySample] = []
    for _ in range(games):
        samples.extend(play_self_play_game(config, mcts, model=model))
    return samples


_pool_shared: dict[str, Any] = {}


def _init_gpu_server_pool(
    request_queue: Any,
    response_queues: list[Any],
    selfplay_kwargs: dict[str, Any],
    mcts_kwargs: dict[str, Any],
) -> None:
    global _pool_shared
    _pool_shared = {
        "request_queue": request_queue,
        "response_queues": response_queues,
        "selfplay_kwargs": selfplay_kwargs,
        "mcts_kwargs": mcts_kwargs,
    }


def _worker_gpu_server_task(args: tuple[int, int]) -> list[SelfPlaySample]:
    worker_id, games = args
    shared = _pool_shared
    config = SelfPlayConfig(**shared["selfplay_kwargs"])
    evaluator = RemoteEvaluator(
        worker_id,
        shared["request_queue"],
        shared["response_queues"][worker_id],
        config.game_config,
    )
    mcts = MCTSConfig(**shared["mcts_kwargs"])
    samples: list[SelfPlaySample] = []
    for _ in range(games):
        samples.extend(play_self_play_game(config, mcts, evaluator=evaluator))
    return samples


def _use_gpu_inference_server(device: str, inference_device: str | None) -> bool:
    infer_dev = inference_device if inference_device is not None else device
    return infer_dev.startswith("cuda")


def run_self_play_parallel(
    checkpoint_path: str | Path,
    selfplay_config: SelfPlayConfig,
    mcts_config: MCTSConfig,
    workers: int = 1,
    *,
    inference_device: str | None = None,
    inference_batch_size: int = 64,
    inference_max_wait_ms: float = 2.0,
) -> list[SelfPlaySample]:
    checkpoint_path = str(checkpoint_path)
    if workers <= 1:
        model, _payload = load_checkpoint(checkpoint_path, map_location=selfplay_config.device)
        return run_self_play(model, selfplay_config, mcts_config)

    games = [selfplay_config.games // workers] * workers
    for idx in range(selfplay_config.games % workers):
        games[idx] += 1
    active_workers = sum(1 for game_count in games if game_count > 0)

    infer_dev = inference_device if inference_device is not None else selfplay_config.device
    ctx = mp.get_context("spawn")
    if _use_gpu_inference_server(selfplay_config.device, inference_device):
        server = start_inference_server(
            checkpoint_path,
            device=infer_dev,
            num_workers=active_workers,
            batch_size=inference_batch_size,
            max_wait_ms=inference_max_wait_ms,
            mp_context=ctx,
        )
        try:
            tasks = [(worker_id, game_count) for worker_id, game_count in enumerate(games) if game_count > 0]
            with ctx.Pool(
                processes=active_workers,
                initializer=_init_gpu_server_pool,
                initargs=(
                    server.request_queue,
                    server.response_queues,
                    selfplay_config.__dict__,
                    mcts_config.__dict__,
                ),
            ) as pool:
                chunks = pool.map(_worker_gpu_server_task, tasks)
        finally:
            server.shutdown()
    else:
        tasks = [
            (
                checkpoint_path,
                selfplay_config.__dict__,
                mcts_config.__dict__,
                game_count,
            )
            for game_count in games
            if game_count > 0
        ]
        with ctx.Pool(processes=active_workers) as pool:
            chunks = pool.map(_worker_local_model, tasks)

    samples: list[SelfPlaySample] = []
    for chunk in chunks:
        samples.extend(chunk)
    return samples
