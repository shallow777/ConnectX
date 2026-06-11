"""表格型 Q-learning (tabular Q-learning), 用于 4x5 connect-3 小棋盘.

标准 6x7 棋盘状态空间约 4.5 万亿, 表格法存不下, 所以这条 baseline 跑在
缩小的棋盘上, 用来展示最基础的 RL 算法以及它的局限性 (作业的算法对比部分)。

关键点 (key ideas):
- 状态做"当前玩家视角"规范化: 自己的棋子恒记为 1, 对手恒记为 2,
  这样双方可以共享同一张 Q 表 (self-play 共用);
- 更新公式是 negamax 风格: target = r - gamma * max_a' Q(s', a'),
  因为 s' 轮到对手行动, 对手的最优 Q 值对自己而言是负收益。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from connectx.agents.lookahead import safe_policy_action
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
    """状态规范化: 把棋盘转成"自己=1, 对手=2"的视角, 双方共享一张 Q 表."""
    opponent = opponent_mark(mark)
    return tuple(1 if cell == mark else 2 if cell == opponent else 0 for cell in board)


@dataclass
class TabularQAgent:
    """表格型 Q-learning agent; q_table: state tuple -> 每列的 Q 值数组."""

    config: ConnectXConfig = field(default_factory=lambda: ConnectXConfig(rows=4, columns=5, inarow=3))
    alpha: float = 0.2    # 学习率 (learning rate)
    gamma: float = 0.99   # 折扣因子 (discount factor)
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
        """Negamax 风格的 TD 更新: 下一状态轮到对手, 对手收益即我方损失."""
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
) -> tuple[TabularQAgent, list[dict[str, float]]]:
    """Self-play 训练: 双方共用一个 agent 轮流落子, 每一步都做 TD 更新.

    返回 (训练好的 agent, 学习曲线), 曲线记录每局的 epsilon / 胜者 / Q 表规模。
    """
    config = config or ConnectXConfig(rows=4, columns=5, inarow=3)
    agent = TabularQAgent(config=config, alpha=alpha, gamma=gamma)
    curve: list[dict[str, float]] = []

    for episode in range(episodes):
        # epsilon 线性退火 (linear decay): 从 epsilon_start 降到 epsilon_end
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - episode / max(episodes - 1, 1))
        board = [0] * (config.rows * config.columns)
        mark = 1
        winner = 0

        for _move in range(config.rows * config.columns):
            action = agent.act(board, mark, epsilon=epsilon, tactical_safety=False)
            before = list(board)
            board = next_board(board, action, mark, config.rows, config.columns)
            done = False
            reward = 0.0
            if check_winner(board, mark, config.rows, config.columns, config.inarow):
                done = True
                winner = mark
                reward = 1.0
            elif is_draw(board, config.rows, config.columns):
                done = True
                winner = 0
                reward = 0.0

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
