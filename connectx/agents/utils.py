from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from connectx.envs.connectx_env import ConnectXConfig, legal_actions


class AgentCallable(Protocol):
    def __call__(self, obs: dict[str, Any], config: ConnectXConfig) -> int:
        ...


def config_value(config: Any, name: str, default: int) -> int:
    if isinstance(config, dict):
        return int(config.get(name, default))
    return int(getattr(config, name, default))


def normalize_config(config: Any) -> ConnectXConfig:
    return ConnectXConfig(
        rows=config_value(config, "rows", 6),
        columns=config_value(config, "columns", 7),
        inarow=config_value(config, "inarow", 4),
    )


def obs_board_mark(obs: Any) -> tuple[list[int], int]:
    if isinstance(obs, dict):
        if "board" in obs:
            return [int(x) for x in obs["board"]], int(obs.get("mark", 1))
        if "action_mask" in obs and "observation" in obs:
            raise ValueError("obs dict does not contain raw board/mark")
    board = getattr(obs, "board")
    mark = getattr(obs, "mark")
    return [int(x) for x in board], int(mark)


def legal_from_obs(obs: dict[str, Any], config: ConnectXConfig) -> list[int]:
    if "action_mask" in obs:
        mask = np.asarray(obs["action_mask"])
        return [int(action) for action, valid in enumerate(mask) if bool(valid)]
    board, _mark = obs_board_mark(obs)
    return legal_actions(board, config.rows, config.columns)


def masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    values[~mask] = -np.inf
    return int(np.argmax(values))


def masked_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    probs = np.zeros_like(logits, dtype=np.float64)
    if not mask.any():
        return probs
    temperature = max(float(temperature), 1e-6)
    valid_logits = logits[mask] / temperature
    valid_logits = valid_logits - np.max(valid_logits)
    exp_logits = np.exp(valid_logits)
    probs[mask] = exp_logits / exp_logits.sum()
    return probs


def sample_masked_action(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> int:
    probs = masked_softmax(logits, mask, temperature)
    if probs.sum() <= 0:
        legal = np.flatnonzero(mask)
        return int(legal[0]) if legal.size else 0
    return int(np.random.choice(np.arange(len(probs)), p=probs))


def center_preferred_action(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    columns = len(mask)
    order = sorted(range(columns), key=lambda col: (abs(col - columns // 2), col))
    for action in order:
        if mask[action]:
            return int(action)
    return 0


def epsilon_greedy_action(q_values: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
    mask = np.asarray(mask, dtype=bool)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return 0
    if np.random.random() < epsilon:
        return int(np.random.choice(legal))
    return masked_argmax(np.asarray(q_values), mask)
