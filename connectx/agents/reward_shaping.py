from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from connectx.agents.lookahead import immediate_winning_actions
from connectx.envs.connectx_env import (
    ConnectXConfig,
    _board_array,
    check_winner,
    is_draw,
    opponent_mark,
)


@dataclass(frozen=True)
class RewardShapingConfig:
    """Potential-style shaping: sparse terminal rewards plus heuristic deltas."""

    enabled: bool = True
    gamma: float = 0.99
    win_reward: float = 1.0
    draw_reward: float = 0.0
    one_in_row: float = 0.005
    two_in_row: float = 0.015
    three_in_row: float = 0.05
    center_bonus: float = 0.01
    immediate_threat: float = 0.08


def count_open_windows(
    board: list[int] | tuple[int, ...] | np.ndarray,
    mark: int,
    config: ConnectXConfig,
) -> dict[int, int]:
    """Count length-inarow windows with only mark stones and empty cells."""
    array = _board_array(board, config.rows, config.columns)
    opponent = opponent_mark(mark)
    counts = {1: 0, 2: 0, 3: 0}
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))

    for row in range(config.rows):
        for col in range(config.columns):
            for dr, dc in directions:
                cells: list[int] = []
                valid = True
                for offset in range(config.inarow):
                    rr = row + offset * dr
                    cc = col + offset * dc
                    if rr < 0 or rr >= config.rows or cc < 0 or cc >= config.columns:
                        valid = False
                        break
                    cells.append(int(array[rr, cc]))
                if not valid:
                    continue
                if opponent in cells:
                    continue
                mark_count = sum(1 for cell in cells if cell == mark)
                empty_count = sum(1 for cell in cells if cell == 0)
                if mark_count <= 0 or mark_count + empty_count != config.inarow:
                    continue
                if mark_count >= config.inarow:
                    continue
                counts[mark_count] = counts.get(mark_count, 0) + 1
    return counts


def position_value(
    board: list[int] | tuple[int, ...] | np.ndarray,
    mark: int,
    config: ConnectXConfig,
    shaping: RewardShapingConfig,
) -> float:
    opponent = opponent_mark(mark)
    own = count_open_windows(board, mark, config)
    opp = count_open_windows(board, opponent, config)

    score = 0.0
    score += shaping.one_in_row * (own.get(1, 0) - opp.get(1, 0))
    score += shaping.two_in_row * (own.get(2, 0) - opp.get(2, 0))
    score += shaping.three_in_row * (own.get(3, 0) - opp.get(3, 0))
    score += shaping.immediate_threat * (
        len(immediate_winning_actions(list(board), mark, config))
        - len(immediate_winning_actions(list(board), opponent, config))
    )

    center_col = config.columns // 2
    array = _board_array(board, config.rows, config.columns)
    for row in range(config.rows):
        cell = int(array[row, center_col])
        if cell == mark:
            score += shaping.center_bonus
        elif cell == opponent:
            score -= shaping.center_bonus * 0.5
    return float(score)


def compute_step_reward(
    before_board: list[int] | tuple[int, ...],
    after_board: list[int] | tuple[int, ...],
    mark: int,
    config: ConnectXConfig,
    shaping: RewardShapingConfig | None,
) -> float:
    if check_winner(after_board, mark, config.rows, config.columns, config.inarow):
        return shaping.win_reward if shaping is not None else 1.0
    if is_draw(after_board, config.rows, config.columns):
        return shaping.draw_reward if shaping is not None else 0.0

    if shaping is None or not shaping.enabled:
        return 0.0

    return position_value(after_board, mark, config, shaping) - position_value(
        before_board, mark, config, shaping
    )


def compute_shaping_delta(
    before_board: list[int] | tuple[int, ...],
    after_board: list[int] | tuple[int, ...],
    mark: int,
    config: ConnectXConfig,
    shaping: RewardShapingConfig | None,
) -> float:
    if shaping is None or not shaping.enabled:
        return 0.0
    if check_winner(after_board, mark, config.rows, config.columns, config.inarow):
        return 0.0
    if is_draw(after_board, config.rows, config.columns):
        return 0.0
    return position_value(after_board, mark, config, shaping) - position_value(
        before_board, mark, config, shaping
    )


def terminal_outcome(sample_mark: int, winner: int) -> float:
    if winner == 0:
        return 0.0
    return 1.0 if winner == sample_mark else -1.0


def discounted_shaping_returns(shaping_rewards: list[float], gamma: float) -> list[float]:
    if not shaping_rewards:
        return []
    returns = [0.0] * len(shaping_rewards)
    for index in range(len(shaping_rewards) - 1, -1, -1):
        if index == len(shaping_rewards) - 1:
            returns[index] = shaping_rewards[index]
        else:
            returns[index] = shaping_rewards[index] + gamma * (-returns[index + 1])
    return returns


def compute_alphazero_value_targets(
    marks: list[int],
    winner: int,
    shaping_rewards: list[float],
    *,
    gamma: float = 0.99,
) -> list[float]:
    shaped = discounted_shaping_returns(shaping_rewards, gamma)
    return [
        float(np.clip(terminal_outcome(mark, winner) + shaped[index], -1.0, 1.0))
        for index, mark in enumerate(marks)
    ]
