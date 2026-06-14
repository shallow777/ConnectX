from __future__ import annotations

from typing import Any, Callable

import numpy as np

from connectx.agents.utils import center_preferred_action, normalize_config, obs_board_mark
from connectx.envs.connectx_env import (
    ConnectXConfig,
    check_winner,
    legal_actions,
    next_board,
    opponent_mark,
    valid_action_mask,
)


def opponent_win_count(
    board: list[int],
    mark: int,
    config: ConnectXConfig,
) -> int:
    opponent = opponent_mark(mark)
    return sum(
        1
        for action in legal_actions(board, config.rows, config.columns)
        if check_winner(
            next_board(board, action, opponent, config.rows, config.columns),
            opponent,
            config.rows,
            config.columns,
            config.inarow,
        )
    )


def opponent_best_reply_threat_count(
    board: list[int],
    mark: int,
    config: ConnectXConfig,
) -> int:
    """Max immediate winning columns for opponent after one opponent reply."""
    opponent = opponent_mark(mark)
    best = 0
    for action in legal_actions(board, config.rows, config.columns):
        after = next_board(board, action, opponent, config.rows, config.columns)
        best = max(best, opponent_win_count(after, mark, config))
    return best


def proactive_defensive_action(
    board: list[int],
    mark: int,
    config: ConnectXConfig,
) -> int | None:
    legal = legal_actions(board, config.rows, config.columns)
    if not legal:
        return None

    def move_score(action: int) -> tuple[int, int, int]:
        after = next_board(board, action, mark, config.rows, config.columns)
        immediate = opponent_win_count(after, mark, config)
        reply = opponent_best_reply_threat_count(after, mark, config)
        col = action
        repeat = 0
        for row in range(config.rows):
            idx = row * config.columns + col
            if board[idx] == mark:
                repeat = config.rows - row
                break
        return immediate, reply, repeat

    best_action = legal[0]
    best_score = move_score(best_action)
    for action in legal[1:]:
        score = move_score(action)
        if score < best_score:
            best_score = score
            best_action = action
    return best_action


def immediate_winning_actions(
    board: list[int],
    mark: int,
    config: ConnectXConfig,
) -> list[int]:
    wins: list[int] = []
    for action in legal_actions(board, config.rows, config.columns):
        candidate = next_board(board, action, mark, config.rows, config.columns)
        if check_winner(candidate, mark, config.rows, config.columns, config.inarow):
            wins.append(action)
    return wins


def best_block_action(
    board: list[int],
    mark: int,
    config: ConnectXConfig,
) -> int | None:
    """Block opponent wins; when multiple threats exist, pick the most defensive column."""
    opponent = opponent_mark(mark)
    threats = immediate_winning_actions(board, opponent, config)
    if not threats:
        return None
    if len(threats) == 1:
        return threats[0]

    best_action = threats[0]
    best_remaining = len(threats)
    for action in threats:
        after = next_board(board, action, mark, config.rows, config.columns)
        remaining = len(immediate_winning_actions(after, opponent, config))
        if remaining < best_remaining:
            best_remaining = remaining
            best_action = action
    return best_action


def tactical_action(
    board: list[int],
    mark: int,
    config: ConnectXConfig,
) -> int | None:
    own_wins = immediate_winning_actions(board, mark, config)
    if own_wins:
        return own_wins[0]

    block = best_block_action(board, mark, config)
    if block is not None:
        return block

    if opponent_best_reply_threat_count(board, mark, config) >= 2:
        return proactive_defensive_action(board, mark, config)

    return None


def safe_policy_action(
    board: list[int],
    mark: int,
    config: ConnectXConfig,
    policy_action: int | None,
) -> int:
    action = tactical_action(board, mark, config)
    if action is not None:
        return action

    mask = valid_action_mask(board, config.rows, config.columns)
    if policy_action is not None and 0 <= int(policy_action) < config.columns and mask[int(policy_action)]:
        return int(policy_action)
    return center_preferred_action(mask)


def wrap_with_tactical_safety(
    policy: Callable[[dict[str, Any], ConnectXConfig], int],
) -> Callable[[dict[str, Any], Any], int]:
    def agent(obs: dict[str, Any], config: Any) -> int:
        normalized = normalize_config(config)
        board, mark = obs_board_mark(obs)
        policy_action = policy(obs, normalized)
        return safe_policy_action(board, mark, normalized, policy_action)

    return agent


def tactical_agent(obs: dict[str, Any], config: Any) -> int:
    normalized = normalize_config(config)
    board, mark = obs_board_mark(obs)
    mask = np.asarray(valid_action_mask(board, normalized.rows, normalized.columns), dtype=bool)
    return safe_policy_action(board, mark, normalized, center_preferred_action(mask))
