"""AlphaZero self-play 数据生成 (单进程与多进程版本).

每个样本是 (encoded_state, mcts_policy, z, action_mask):
- mcts_policy 是该步 MCTS 访问次数归一化得到的分布 (policy 的训练目标);
- z 是终局结果回填到每一步 (该步行动方赢 +1 / 输 -1 / 平 0, value 的训练目标)。
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from connectx.agents.alphazero.mcts import AlphaZeroMCTS, MCTSConfig, TorchEvaluator
from connectx.agents.alphazero.network import AlphaZeroNet, load_checkpoint
from connectx.envs.connectx_env import (
    ConnectXConfig,
    check_winner,
    encode_board,
    is_draw,
    next_board,
    opponent_mark,
    valid_action_mask,
)


SelfPlaySample = tuple[np.ndarray, np.ndarray, float, np.ndarray]


@dataclass(frozen=True)
class SelfPlayConfig:
    games: int = 20
    max_moves: int = 42
    temperature_moves: int = 8  # 前 N 步用 T=1 采样 (开局多样性), 之后 argmax
    rows: int = 6
    columns: int = 7
    inarow: int = 4
    device: str = "cpu"

    @property
    def game_config(self) -> ConnectXConfig:
        return ConnectXConfig(rows=self.rows, columns=self.columns, inarow=self.inarow)


def play_self_play_game(
    model: AlphaZeroNet,
    selfplay_config: SelfPlayConfig,
    mcts_config: MCTSConfig,
) -> list[SelfPlaySample]:
    """下一整局 self-play, 双方都由同一模型 + MCTS 控制, 返回带标签的样本."""
    config = selfplay_config.game_config
    evaluator = TorchEvaluator(model, config, device=selfplay_config.device)
    mcts = AlphaZeroMCTS(evaluator, config, mcts_config)

    board = [0] * (config.rows * config.columns)
    mark = 1
    history: list[tuple[np.ndarray, np.ndarray, int, np.ndarray]] = []
    winner = 0

    for move_idx in range(selfplay_config.max_moves):
        # 前 temperature_moves 步带温度采样, 之后近似 argmax (T->0)
        temperature = 1.0 if move_idx < selfplay_config.temperature_moves else 1e-6
        mask = valid_action_mask(board, config.rows, config.columns)
        encoded = encode_board(board, mark, config.rows, config.columns)
        policy = mcts.search(board, mark, add_noise=True, temperature=temperature)
        if policy.sum() <= 0:
            legal = np.flatnonzero(mask)
            action = int(np.random.choice(legal))
        else:
            action = int(np.random.choice(np.arange(config.columns), p=policy / policy.sum()))

        history.append((encoded, policy.astype(np.float32), mark, mask.astype(bool)))
        board = next_board(board, action, mark, config.rows, config.columns)

        if check_winner(board, mark, config.rows, config.columns, config.inarow):
            winner = mark
            break
        if is_draw(board, config.rows, config.columns):
            winner = 0
            break
        mark = opponent_mark(mark)

    # 终局结果 z 回填到每一步 (从该步行动方视角): 胜 +1 / 负 -1 / 平 0
    samples: list[SelfPlaySample] = []
    for encoded, policy, sample_mark, mask in history:
        if winner == 0:
            z = 0.0
        else:
            z = 1.0 if winner == sample_mark else -1.0
        samples.append((encoded, policy, z, mask))
    return samples


def run_self_play(
    model: AlphaZeroNet,
    selfplay_config: SelfPlayConfig,
    mcts_config: MCTSConfig,
) -> list[SelfPlaySample]:
    samples: list[SelfPlaySample] = []
    for _ in range(selfplay_config.games):
        samples.extend(play_self_play_game(model, selfplay_config, mcts_config))
    return samples


def _worker(args: tuple[str, dict[str, Any], dict[str, Any], int]) -> list[SelfPlaySample]:
    """子进程入口: 每个 worker 独立从 checkpoint 加载模型再自我对弈."""
    checkpoint_path, selfplay_kwargs, mcts_kwargs, games = args
    model, _payload = load_checkpoint(checkpoint_path, map_location=selfplay_kwargs.get("device", "cpu"))
    config = SelfPlayConfig(**{**selfplay_kwargs, "games": games})
    return run_self_play(model, config, MCTSConfig(**mcts_kwargs))


def run_self_play_parallel(
    checkpoint_path: str | Path,
    selfplay_config: SelfPlayConfig,
    mcts_config: MCTSConfig,
    workers: int = 1,
) -> list[SelfPlaySample]:
    """多进程 self-play: 把总局数尽量均分给各 worker (spawn 模式, CUDA 安全)."""
    checkpoint_path = str(checkpoint_path)
    if workers <= 1:
        model, _payload = load_checkpoint(checkpoint_path, map_location=selfplay_config.device)
        return run_self_play(model, selfplay_config, mcts_config)

    games = [selfplay_config.games // workers] * workers
    for idx in range(selfplay_config.games % workers):
        games[idx] += 1
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
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        chunks = pool.map(_worker, tasks)
    samples: list[SelfPlaySample] = []
    for chunk in chunks:
        samples.extend(chunk)
    return samples
