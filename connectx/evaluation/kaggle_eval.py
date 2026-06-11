from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from connectx.agents.utils import normalize_config
from connectx.envs.connectx_env import ConnectXConfig


KaggleAgent = Callable[[Any, Any], int]


@dataclass(frozen=True)
class KaggleEvalResult:
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    mean_reward: float
    raw_rewards: list[float]


def to_kaggle_agent(agent: Callable[[dict[str, Any], ConnectXConfig], int]) -> KaggleAgent:
    def wrapped(obs: Any, config: Any) -> int:
        if isinstance(obs, dict):
            board = [int(x) for x in obs["board"]]
            mark = int(obs["mark"])
        else:
            board = [int(x) for x in obs.board]
            mark = int(obs.mark)
        normalized = normalize_config(config)
        return int(agent({"board": board, "mark": mark}, normalized))

    return wrapped


def evaluate_against_negamax(
    agent: Callable[[dict[str, Any], ConnectXConfig], int],
    *,
    games: int = 20,
    rows: int = 6,
    columns: int = 7,
    inarow: int = 4,
    timeout: float = 2.0,
) -> KaggleEvalResult:
    try:
        from kaggle_environments import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "kaggle_environments is required for evaluate_against_negamax; "
            "install the optional kaggle dependency on the server."
        ) from exc

    config = {"rows": rows, "columns": columns, "inarow": inarow, "timeout": timeout}
    wrapped = to_kaggle_agent(agent)
    rewards = evaluate(
        "connectx",
        [wrapped, "negamax"],
        configuration=config,
        num_episodes=games,
    )
    # rewards 每项是 [我方, negamax] 的对局结果; 我方排第一位, 故取 pair[0]
    first_player_rewards = [float(pair[0]) for pair in rewards]
    wins = sum(reward > 0 for reward in first_player_rewards)
    losses = sum(reward < 0 for reward in first_player_rewards)
    draws = games - wins - losses
    mean_reward = float(np.mean(first_player_rewards)) if first_player_rewards else 0.0
    return KaggleEvalResult(
        games=games,
        wins=int(wins),
        losses=int(losses),
        draws=int(draws),
        win_rate=float(wins / games) if games else 0.0,
        mean_reward=mean_reward,
        raw_rewards=first_player_rewards,
    )
