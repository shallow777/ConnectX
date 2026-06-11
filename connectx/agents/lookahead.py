"""战术安全层 (tactical safety layer): 一步前瞻的必胜/必堵规则.

所有学习型 agent (Q-learning / DQN / PPO / AlphaZero) 都可以套这一层:
1. 自己有一步制胜的列 -> 直接下 (take the immediate win);
2. 对手下一步能赢 -> 立刻堵住 (block the immediate loss);
3. 否则才采用策略网络给出的动作。
这能消除神经网络偶尔漏看一步杀招导致的低级失误, 对 Kaggle 分数提升明显。
"""

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


def immediate_winning_actions(
    board: list[int],
    mark: int,
    config: ConnectXConfig,
) -> list[int]:
    """枚举 mark 一步即可获胜的所有列 (columns that win immediately)."""
    wins: list[int] = []
    for action in legal_actions(board, config.rows, config.columns):
        candidate = next_board(board, action, mark, config.rows, config.columns)
        if check_winner(candidate, mark, config.rows, config.columns, config.inarow):
            wins.append(action)
    return wins


def tactical_action(
    board: list[int],
    mark: int,
    config: ConnectXConfig,
) -> int | None:
    """优先级: 先拿下自己的必胜点, 再堵对手的必胜点; 都没有则返回 None."""
    own_wins = immediate_winning_actions(board, mark, config)
    if own_wins:
        return own_wins[0]

    opponent = opponent_mark(mark)
    blocks = immediate_winning_actions(board, opponent, config)
    if blocks:
        return blocks[0]

    return None


def safe_policy_action(
    board: list[int],
    mark: int,
    config: ConnectXConfig,
    policy_action: int | None,
) -> int:
    """战术规则优先, 然后采纳策略动作; 策略动作非法时回退到中心偏好列."""
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
    """纯规则 baseline: 只用战术安全层 + 中心偏好, 不依赖任何学习模型."""
    normalized = normalize_config(config)
    board, mark = obs_board_mark(obs)
    mask = np.asarray(valid_action_mask(board, normalized.rows, normalized.columns), dtype=bool)
    return safe_policy_action(board, mark, normalized, center_preferred_action(mask))
