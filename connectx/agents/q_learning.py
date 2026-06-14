from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from connectx.agents.lookahead import safe_policy_action
from connectx.agents.reward_shaping import RewardShapingConfig, compute_step_reward
from connectx.agents.utils import epsilon_greedy_action
from connectx.envs.connectx_env import (
    ConnectXConfig,
    check_winner,
    is_draw,
    next_board,
    opponent_mark,
    valid_action_mask,
)


def canonical_state(board: list[int] | tuple[int, ...], mark: int) -> tuple[int, ...]:
    opponent = opponent_mark(mark)
    return tuple(1 if cell == mark else 2 if cell == opponent else 0 for cell in board)


@dataclass
class TabularQAgent:
    config: ConnectXConfig = field(default_factory=lambda: ConnectXConfig(rows=4, columns=5, inarow=3))
    alpha: float = 0.2
    gamma: float = 0.99
    q_table: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)

    def values(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.config.columns, dtype=np.float32)
        return self.q_table[state]

    def act(self, board: list[int], mark: int, epsilon: float = 0.0, tactical_safety: bool = False) -> int:
        state = canonical_state(board, mark)
        mask = valid_action_mask(board, self.config.rows, self.config.columns).astype(bool)
        action = epsilon_greedy_action(self.values(state), mask, epsilon)
        if tactical_safety:
            action = safe_policy_action(board, mark, self.config, action)
        return int(action)

    def update(self, board: list[int], mark: int, action: int, reward: float, next_board_state: list[int], done: bool) -> None:
        state = canonical_state(board, mark)
        q_values = self.values(state)
        if done:
            target = reward
        else:
            next_mark = opponent_mark(mark)
            next_state = canonical_state(next_board_state, next_mark)
            next_mask = valid_action_mask(next_board_state, self.config.rows, self.config.columns).astype(bool)
            opponent_best = float(self.values(next_state)[next_mask].max()) if next_mask.any() else 0.0
            target = reward - self.gamma * opponent_best
        q_values[action] += self.alpha * (float(target) - float(q_values[action]))

    def agent_fn(self, tactical_safety: bool = True):
        def agent(obs: dict[str, Any], config: ConnectXConfig) -> int:
            del config
            board = [int(x) for x in obs["board"]]
            mark = int(obs["mark"])
            return self.act(board, mark, epsilon=0.0, tactical_safety=tactical_safety)

        return agent

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"config": self.config, "alpha": self.alpha, "gamma": self.gamma, "q_table": self.q_table}, f)

    @classmethod
    def load(cls, path: str | Path) -> "TabularQAgent":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        return cls(
            config=payload["config"],
            alpha=float(payload["alpha"]),
            gamma=float(payload["gamma"]),
            q_table=payload["q_table"],
        )


def train_tabular_q_learning(
    episodes: int,
    *,
    config: ConnectXConfig | None = None,
    alpha: float = 0.2,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    reward_shaping: RewardShapingConfig | None = None,
) -> tuple[TabularQAgent, list[dict[str, float]]]:
    config = config or ConnectXConfig(rows=4, columns=5, inarow=3)
    agent = TabularQAgent(config=config, alpha=alpha, gamma=gamma)
    curve: list[dict[str, float]] = []

    for episode in range(episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - episode / max(episodes - 1, 1))
        board = [0] * (config.rows * config.columns)
        mark = 1
        winner = 0

        for _move in range(config.rows * config.columns):
            action = agent.act(board, mark, epsilon=epsilon, tactical_safety=False)
            before = list(board)
            board = next_board(board, action, mark, config.rows, config.columns)
            reward = compute_step_reward(before, board, mark, config, reward_shaping)
            done = False
            if check_winner(board, mark, config.rows, config.columns, config.inarow):
                done = True
                winner = mark
            elif is_draw(board, config.rows, config.columns):
                done = True
                winner = 0

            agent.update(before, mark, action, reward, board, done)
            if done:
                break
            mark = opponent_mark(mark)

        curve.append(
            {
                "episode": float(episode),
                "epsilon": float(epsilon),
                "winner": float(winner),
                "q_states": float(len(agent.q_table)),
            }
        )

    return agent, curve
