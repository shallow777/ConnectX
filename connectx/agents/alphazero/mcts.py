from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from connectx.agents.alphazero.network import AlphaZeroNet, predict_policy_value, predict_policy_value_batch
from connectx.envs.connectx_env import (
    ConnectXConfig,
    check_winner,
    encode_board,
    is_draw,
    next_board,
    opponent_mark,
    valid_action_mask,
)


Evaluator = Callable[[tuple[int, ...], int], tuple[np.ndarray, float]]


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 100
    eval_batch_size: int = 16
    c_puct: float = 2.0
    dirichlet_alpha: float = 1.0
    dirichlet_eps: float = 0.25


@dataclass
class SearchNode:
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
        simulations: int | None = None,
    ) -> np.ndarray:
        root = SearchNode(prior=1.0, to_play=mark)
        board_tuple = tuple(int(x) for x in board)
        self._expand(root, board_tuple, mark)
        if add_noise:
            self._add_dirichlet_noise(root)

        sim_budget = self.mcts_config.simulations if simulations is None else int(simulations)
        sims_done = 0
        batch_size = max(1, self.mcts_config.eval_batch_size)
        while sims_done < sim_budget:
            chunk = min(batch_size, sim_budget - sims_done)
            pending: list[tuple[SearchNode, tuple[int, ...], int, list[SearchNode]]] = []
            for _ in range(chunk):
                node = root
                scratch_board = board_tuple
                to_play = mark
                path = [node]

                while node.expanded():
                    action, node = self._select_child(node)
                    scratch_board = tuple(
                        next_board(scratch_board, action, to_play, self.config.rows, self.config.columns)
                    )
                    to_play = opponent_mark(to_play)
                    path.append(node)

                terminal_value = self._terminal_value(scratch_board, to_play)
                if terminal_value is not None:
                    self._backpropagate(path, terminal_value)
                else:
                    pending.append((node, scratch_board, to_play, path))

            if pending:
                values = self._batch_expand(
                    [item[0] for item in pending],
                    [item[1] for item in pending],
                    [item[2] for item in pending],
                )
                for (_, _, _, path), value in zip(pending, values):
                    self._backpropagate(path, value)

            sims_done += chunk

        return self._visit_policy(root, temperature)

    def _select_child(self, node: SearchNode) -> tuple[int, SearchNode]:
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

    def _terminal_value(self, board: tuple[int, ...], to_play: int) -> float | None:
        previous = opponent_mark(to_play)
        if check_winner(board, previous, self.config.rows, self.config.columns, self.config.inarow):
            return -1.0
        if is_draw(board, self.config.rows, self.config.columns):
            return 0.0
        return None

    def _evaluate_leaf(self, node: SearchNode, board: tuple[int, ...], to_play: int) -> float:
        terminal_value = self._terminal_value(board, to_play)
        if terminal_value is not None:
            return terminal_value
        return self._expand(node, board, to_play)

    def _batch_expand(
        self,
        nodes: list[SearchNode],
        boards: list[tuple[int, ...]],
        to_plays: list[int],
    ) -> list[float]:
        evaluate_batch = getattr(self.evaluator, "evaluate_batch", None)
        if evaluate_batch is not None and len(nodes) > 1:
            results = evaluate_batch(boards, to_plays)
            values: list[float] = []
            for node, (priors, value), board, to_play in zip(nodes, results, boards, to_plays):
                mask = valid_action_mask(board, self.config.rows, self.config.columns).astype(bool)
                for action, valid in enumerate(mask):
                    if valid:
                        node.children[action] = SearchNode(
                            prior=float(priors[action]),
                            to_play=opponent_mark(to_play),
                        )
                values.append(float(value))
            return values

        return [self._expand(node, board, to_play) for node, board, to_play in zip(nodes, boards, to_plays)]

    def _expand(self, node: SearchNode, board: tuple[int, ...], to_play: int) -> float:
        priors, value = self.evaluator(board, to_play)
        mask = valid_action_mask(board, self.config.rows, self.config.columns).astype(bool)
        for action, valid in enumerate(mask):
            if valid:
                node.children[action] = SearchNode(prior=float(priors[action]), to_play=opponent_mark(to_play))
        return float(value)

    def _backpropagate(self, path: list[SearchNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _visit_policy(self, root: SearchNode, temperature: float) -> np.ndarray:
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
        actions = list(root.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.mcts_config.dirichlet_alpha] * len(actions))
        for action, eta in zip(actions, noise):
            child = root.children[action]
            child.prior = (1.0 - self.mcts_config.dirichlet_eps) * child.prior + self.mcts_config.dirichlet_eps * float(eta)


class TorchEvaluator:
    def __init__(self, model: AlphaZeroNet, config: ConnectXConfig | None = None, device: str = "cpu") -> None:
        self.model = model
        self.config = config or ConnectXConfig(rows=model.rows, columns=model.columns)
        self.device = device

    def __call__(self, board: tuple[int, ...], mark: int) -> tuple[np.ndarray, float]:
        encoded = encode_board(board, mark, self.config.rows, self.config.columns)
        mask = valid_action_mask(board, self.config.rows, self.config.columns)
        return predict_policy_value(self.model, encoded, mask, self.device)

    def evaluate_batch(
        self,
        boards: list[tuple[int, ...]],
        marks: list[int],
    ) -> list[tuple[np.ndarray, float]]:
        if not boards:
            return []
        encoded = np.stack(
            [
                encode_board(board, mark, self.config.rows, self.config.columns)
                for board, mark in zip(boards, marks)
            ],
            axis=0,
        )
        masks = np.stack(
            [
                valid_action_mask(board, self.config.rows, self.config.columns)
                for board in boards
            ],
            axis=0,
        )
        policies, values = predict_policy_value_batch(self.model, encoded, masks, self.device)
        return [(policies[i], float(values[i])) for i in range(len(boards))]
