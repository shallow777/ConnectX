"""PUCT 蒙特卡洛树搜索 (MCTS), AlphaZero 的核心搜索组件.

约定 (sign convention): 节点的 value 始终是"该节点行动方"视角的期望收益,
所以父节点选子时用 -child.value (对手的好局面就是我的坏局面),
反向传播时每上一层符号取反一次 (negamax-style backup)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from connectx.agents.alphazero.network import AlphaZeroNet, predict_policy_value
from connectx.envs.connectx_env import (
    ConnectXConfig,
    check_winner,
    encode_board,
    is_draw,
    next_board,
    opponent_mark,
    valid_action_mask,
)


# 评估函数: (board, mark) -> (各列先验概率 priors, 局面价值 value)
Evaluator = Callable[[tuple[int, ...], int], tuple[np.ndarray, float]]


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 100        # 每步的模拟次数 (越多越强越慢)
    c_puct: float = 2.0           # 探索系数, 平衡先验与访问次数
    dirichlet_alpha: float = 1.0  # 根节点 Dirichlet 噪声参数 (self-play 探索用)
    dirichlet_eps: float = 0.25   # 噪声混合比例


@dataclass
class SearchNode:
    """搜索树节点; value 为 to_play 一方视角的平均价值."""

    prior: float
    to_play: int
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return bool(self.children)

    def puct_score(self, parent_visits: int) -> float:
        prior_score = math.sqrt(parent_visits + 1) * self.prior / (1 + self.visit_count)
        return -self.value + prior_score


class AlphaZeroMCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        config: ConnectXConfig | None = None,
        mcts_config: MCTSConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or ConnectXConfig()
        self.mcts_config = mcts_config or MCTSConfig()

    def search(
        self,
        board: list[int] | tuple[int, ...],
        mark: int,
        *,
        add_noise: bool = False,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """跑 simulations 次模拟, 返回根节点访问次数构成的策略分布.

        add_noise 只在 self-play 时打开 (增加探索); 评估/实战时关闭。
        每次模拟分三步: select (PUCT 下行) -> expand & evaluate -> backpropagate。
        """
        root = SearchNode(prior=1.0, to_play=mark)
        board_tuple = tuple(int(x) for x in board)
        self._expand(root, board_tuple, mark)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.mcts_config.simulations):
            node = root
            scratch_board = board_tuple
            to_play = mark
            path = [node]

            # Select: 沿 PUCT 分数最高的孩子一路下行到叶节点
            while node.expanded():
                action, node = self._select_child(node)
                scratch_board = tuple(
                    next_board(scratch_board, action, to_play, self.config.rows, self.config.columns)
                )
                to_play = opponent_mark(to_play)
                path.append(node)

            value = self._evaluate_leaf(node, scratch_board, to_play)
            self._backpropagate(path, value)

        return self._visit_policy(root, temperature)

    def _select_child(self, node: SearchNode) -> tuple[int, SearchNode]:
        """按 PUCT 公式选孩子: score = -Q(child) + c_puct * sqrt(N_parent) * P / (1 + N_child).

        -Q 是因为 child.value 是对手视角; 第二项是先验加权的探索奖励 (UCB)。
        """
        parent_visits = max(node.visit_count, 1)
        best_action = -1
        best_child: SearchNode | None = None
        best_score = -float("inf")
        for action, child in node.children.items():
            score = -child.value + self.mcts_config.c_puct * math.sqrt(parent_visits) * child.prior / (1 + child.visit_count)
            if score > best_score:
                best_action = action
                best_child = child
                best_score = score
        if best_child is None:
            raise RuntimeError("Cannot select a child from an unexpanded node")
        return best_action, best_child

    def _evaluate_leaf(self, node: SearchNode, board: tuple[int, ...], to_play: int) -> float:
        """叶节点价值: 终局直接给真值, 否则展开并用网络估值 (to_play 视角)."""
        previous = opponent_mark(to_play)
        # 上一手 (对手) 刚连成线 => 当前行动方已输
        if check_winner(board, previous, self.config.rows, self.config.columns, self.config.inarow):
            return -1.0
        if is_draw(board, self.config.rows, self.config.columns):
            return 0.0
        return self._expand(node, board, to_play)

    def _expand(self, node: SearchNode, board: tuple[int, ...], to_play: int) -> float:
        priors, value = self.evaluator(board, to_play)
        mask = valid_action_mask(board, self.config.rows, self.config.columns).astype(bool)
        for action, valid in enumerate(mask):
            if valid:
                node.children[action] = SearchNode(prior=float(priors[action]), to_play=opponent_mark(to_play))
        return float(value)

    def _backpropagate(self, path: list[SearchNode], value: float) -> None:
        """自底向上回传价值; 每上一层视角切换一次, 故符号取反 (negamax backup)."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _visit_policy(self, root: SearchNode, temperature: float) -> np.ndarray:
        """把根节点孩子的访问次数转成策略: pi(a) ∝ N(a)^(1/T); T→0 退化为 argmax."""
        visits = np.zeros(self.config.columns, dtype=np.float64)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if visits.sum() <= 0:
            mask = np.asarray([action in root.children for action in range(self.config.columns)], dtype=bool)
            visits[mask] = 1.0
        if temperature <= 1e-6:
            policy = np.zeros_like(visits)
            policy[int(np.argmax(visits))] = 1.0
            return policy.astype(np.float32)
        visits = np.power(visits, 1.0 / temperature)
        policy = visits / visits.sum()
        return policy.astype(np.float32)

    def _add_dirichlet_noise(self, root: SearchNode) -> None:
        """根节点先验混入 Dirichlet 噪声: P' = (1-eps)*P + eps*eta, self-play 探索用."""
        actions = list(root.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.mcts_config.dirichlet_alpha] * len(actions))
        for action, eta in zip(actions, noise):
            child = root.children[action]
            child.prior = (1.0 - self.mcts_config.dirichlet_eps) * child.prior + self.mcts_config.dirichlet_eps * float(eta)


class TorchEvaluator:
    """把 AlphaZeroNet 包装成 MCTS 需要的 Evaluator 接口."""

    def __init__(self, model: AlphaZeroNet, config: ConnectXConfig | None = None, device: str = "cpu") -> None:
        self.model = model
        self.config = config or ConnectXConfig(rows=model.rows, columns=model.columns)
        self.device = device

    def __call__(self, board: tuple[int, ...], mark: int) -> tuple[np.ndarray, float]:
        encoded = encode_board(board, mark, self.config.rows, self.config.columns)
        mask = valid_action_mask(board, self.config.rows, self.config.columns)
        return predict_policy_value(self.model, encoded, mask, self.device)
