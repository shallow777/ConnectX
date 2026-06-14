from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from connectx.agents.lookahead import safe_policy_action
from connectx.agents.reward_shaping import RewardShapingConfig, compute_step_reward
from connectx.agents.utils import center_preferred_action, masked_argmax
from connectx.envs.connectx_env import (
    ConnectXConfig,
    ConnectXEnv,
    encode_board,
    valid_action_mask,
)


OpponentFn = Callable[[dict[str, Any], ConnectXConfig], int]


def center_opponent(obs: dict[str, Any], config: ConnectXConfig) -> int:
    mask = np.asarray(obs["action_mask"], dtype=bool)
    return center_preferred_action(mask)


def random_opponent(obs: dict[str, Any], config: ConnectXConfig) -> int:
    legal = np.flatnonzero(np.asarray(obs["action_mask"], dtype=bool))
    if legal.size == 0:
        return 0
    return int(np.random.choice(legal))


@dataclass
class OpponentPool:
    opponents: list[OpponentFn] = field(default_factory=lambda: [center_opponent, random_opponent])
    probabilities: list[float] | None = None

    def sample(self) -> OpponentFn:
        if not self.opponents:
            return center_opponent
        if self.probabilities is None:
            return self.opponents[int(np.random.randint(len(self.opponents)))]
        probs = np.asarray(self.probabilities, dtype=np.float64)
        probs = probs / probs.sum()
        return self.opponents[int(np.random.choice(np.arange(len(self.opponents)), p=probs))]

    def add_sb3_model(self, model_path: str | Path, deterministic: bool = True) -> None:
        def opponent(obs: dict[str, Any], config: ConnectXConfig) -> int:
            del config
            try:
                from sb3_contrib import MaskablePPO
            except ImportError as exc:
                raise RuntimeError("sb3-contrib is required to load PPO opponents") from exc

            if not hasattr(opponent, "_model"):
                opponent._model = MaskablePPO.load(str(model_path))  # type: ignore[attr-defined]
            model = opponent._model  # type: ignore[attr-defined]
            model_obs = {
                "observation": obs["observation"],
                "action_mask": obs["action_mask"],
            }
            action, _state = model.predict(
                model_obs,
                deterministic=deterministic,
                action_masks=np.asarray(model_obs["action_mask"], dtype=bool),
            )
            return int(action)

        self.opponents.append(opponent)


class SelfPlayConnectXEnv(gym.Env):
    """Single-agent ConnectX environment for PPO self-play.

    The learning agent controls one mark for the whole episode. After every
    learner move, the sampled opponent responds immediately unless the episode
    has ended. Observations are always encoded from the learner mark's view.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_pool: OpponentPool | None = None,
        rows: int = 6,
        columns: int = 7,
        inarow: int = 4,
        randomize_player: bool = True,
        tactical_safety: bool = True,
        reward_shaping: RewardShapingConfig | None = None,
    ) -> None:
        super().__init__()
        self.base_env = ConnectXEnv(rows=rows, columns=columns, inarow=inarow)
        self.config = self.base_env.config
        self.opponent_pool = opponent_pool or OpponentPool()
        self.randomize_player = randomize_player
        self.tactical_safety = tactical_safety
        self.reward_shaping = reward_shaping

        self.action_space = spaces.Discrete(columns)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2, rows, columns),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(low=0, high=1, shape=(columns,), dtype=np.int8),
            }
        )
        self.learner_mark = 1
        self.opponent: OpponentFn = center_opponent

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.base_env.reset(seed=seed)
        self.learner_mark = int(np.random.choice([1, 2])) if self.randomize_player else 1
        self.opponent = self.opponent_pool.sample()

        if self.learner_mark == 2:
            self._opponent_turn()

        return self._learner_observation(), self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.base_env.done:
            raise RuntimeError("Cannot call step() after termination. Call reset() first.")
        if self.base_env.current_mark != self.learner_mark:
            self._opponent_turn()

        selected = int(action)
        if self.tactical_safety:
            selected = safe_policy_action(
                self.base_env.board,
                self.learner_mark,
                self.config,
                selected,
            )

        before = list(self.base_env.board)
        _obs, _reward, terminated, truncated, info = self.base_env.step(selected)
        if terminated or truncated:
            return self._learner_observation(), self._terminal_reward(), terminated, truncated, self._info(**info)

        after_learner = list(self.base_env.board)
        step_reward = compute_step_reward(
            before,
            after_learner,
            self.learner_mark,
            self.config,
            self.reward_shaping,
        )

        self._opponent_turn()
        if self.base_env.done:
            return self._learner_observation(), self._terminal_reward(), True, False, self._info()

        return self._learner_observation(), step_reward, False, False, self._info()

    def action_masks(self) -> np.ndarray:
        return valid_action_mask(self.base_env.board, self.config.rows, self.config.columns).astype(bool)

    def render(self) -> str:
        return self.base_env.render()

    def _opponent_turn(self) -> None:
        if self.base_env.done or self.base_env.current_mark == self.learner_mark:
            return
        obs = self._mark_observation(self.base_env.current_mark)
        action = self.opponent(obs, self.config)
        if self.tactical_safety:
            action = safe_policy_action(self.base_env.board, self.base_env.current_mark, self.config, action)
        self.base_env.step(int(action))

    def _terminal_reward(self) -> float:
        if not self.base_env.done:
            return 0.0
        if self.base_env.winner == 0:
            return 0.0
        return 1.0 if self.base_env.winner == self.learner_mark else -1.0

    def _mark_observation(self, mark: int) -> dict[str, Any]:
        return {
            "observation": encode_board(self.base_env.board, mark, self.config.rows, self.config.columns),
            "action_mask": valid_action_mask(self.base_env.board, self.config.rows, self.config.columns),
            "board": list(self.base_env.board),
            "mark": mark,
        }

    def _learner_observation(self) -> dict[str, np.ndarray]:
        raw = self._mark_observation(self.learner_mark)
        return {
            "observation": raw["observation"],
            "action_mask": raw["action_mask"],
        }

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = {
            "board": list(self.base_env.board),
            "learner_mark": self.learner_mark,
            "current_mark": self.base_env.current_mark,
            "winner": self.base_env.winner,
            "legal_actions": self.base_env.available_actions(),
        }
        info.update(extra)
        return info


def make_sb3_ppo_agent(model_path: str | Path, deterministic: bool = True) -> Callable[[dict[str, Any], ConnectXConfig], int]:
    def agent(obs: dict[str, Any], config: ConnectXConfig) -> int:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise RuntimeError("sb3-contrib is required to load MaskablePPO agents") from exc

        if not hasattr(agent, "_model"):
            agent._model = MaskablePPO.load(str(model_path))  # type: ignore[attr-defined]
        model = agent._model  # type: ignore[attr-defined]
        if "observation" in obs and "action_mask" in obs:
            model_obs = {"observation": obs["observation"], "action_mask": obs["action_mask"]}
        else:
            board = [int(x) for x in obs["board"]]
            mark = int(obs["mark"])
            model_obs = {
                "observation": encode_board(board, mark, config.rows, config.columns),
                "action_mask": valid_action_mask(board, config.rows, config.columns),
            }
        action, _state = model.predict(
            model_obs,
            deterministic=deterministic,
            action_masks=np.asarray(model_obs["action_mask"], dtype=bool),
        )
        return int(action)

    return agent


def logits_agent(logits_fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[dict[str, Any], ConnectXConfig], int]:
    def agent(obs: dict[str, Any], config: ConnectXConfig) -> int:
        del config
        logits = logits_fn(np.asarray(obs["observation"], dtype=np.float32))
        return masked_argmax(logits, np.asarray(obs["action_mask"], dtype=bool))

    return agent
