"""PPO self-play 所需的对手池与单智能体环境包装 (opponent pool & env wrapper).

MaskablePPO 是单智能体算法, 而 ConnectX 是双人博弈, 所以这里把
"对手怎么下"折叠进环境: 学习方每走一步, 环境内部立刻让对手 (从
OpponentPool 采样) 走一步, 对外表现成一个普通的单智能体 Gym 环境。
对手池里可以不断加入历史 PPO checkpoint, 形成联盟自我对弈 (league self-play)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from connectx.agents.lookahead import safe_policy_action
from connectx.agents.utils import center_preferred_action, masked_argmax
from connectx.envs.connectx_env import (
    ConnectXConfig,
    ConnectXEnv,
    encode_board,
    valid_action_mask,
)


OpponentFn = Callable[[dict[str, Any], ConnectXConfig], int]


def center_opponent(obs: dict[str, Any], config: ConnectXConfig) -> int:
    """规则对手: 永远优先下中间列 (center-first heuristic)."""
    mask = np.asarray(obs["action_mask"], dtype=bool)
    return center_preferred_action(mask)


def random_opponent(obs: dict[str, Any], config: ConnectXConfig) -> int:
    """规则对手: 在合法列中均匀随机落子."""
    legal = np.flatnonzero(np.asarray(obs["action_mask"], dtype=bool))
    if legal.size == 0:
        return 0
    return int(np.random.choice(legal))


@dataclass
class OpponentPool:
    """对手池: 初始只有两个规则对手, 训练中可不断加入冻结的 PPO checkpoint."""

    opponents: list[OpponentFn] = field(default_factory=lambda: [center_opponent, random_opponent])
    probabilities: list[float] | None = None  # None 表示均匀采样

    def sample(self) -> OpponentFn:
        if not self.opponents:
            return center_opponent
        if self.probabilities is None:
            return self.opponents[int(np.random.randint(len(self.opponents)))]
        probs = np.asarray(self.probabilities, dtype=np.float64)
        probs = probs / probs.sum()
        return self.opponents[int(np.random.choice(np.arange(len(self.opponents)), p=probs))]

    def add_sb3_model(self, model_path: str | Path, deterministic: bool = True) -> None:
        """把一个 MaskablePPO checkpoint 冻结后加入对手池 (模型懒加载)."""

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
    """给 PPO 用的单智能体 ConnectX 环境 (single-agent wrapper for self-play).

    学习方整局固定执一种棋子 (reset 时随机先后手); 学习方每走一步,
    环境内部立即让采样到的对手回应一步, 除非对局已结束。
    观测始终从学习方视角编码; 奖励只在终局给出: 胜 +1 / 负 -1 / 平 0。
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
    ) -> None:
        super().__init__()
        self.base_env = ConnectXEnv(rows=rows, columns=columns, inarow=inarow)
        self.config = self.base_env.config
        self.opponent_pool = opponent_pool or OpponentPool()
        self.randomize_player = randomize_player
        self.tactical_safety = tactical_safety

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
        # 随机先后手 + 每局重新采样对手, 保证训练数据多样性
        self.learner_mark = int(np.random.choice([1, 2])) if self.randomize_player else 1
        self.opponent = self.opponent_pool.sample()

        # 学习方执后手时, 先让对手走第一步
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

        _obs, _reward, terminated, truncated, info = self.base_env.step(selected)
        if terminated or truncated:
            return self._learner_observation(), self._terminal_reward(), terminated, truncated, self._info(**info)

        self._opponent_turn()
        terminated = self.base_env.done
        return self._learner_observation(), self._terminal_reward() if terminated else 0.0, terminated, False, self._info()

    def action_masks(self) -> np.ndarray:
        return valid_action_mask(self.base_env.board, self.config.rows, self.config.columns).astype(bool)

    def render(self) -> str:
        return self.base_env.render()

    def _opponent_turn(self) -> None:
        """轮到对手时让对手走一步 (终局或不该对手走时直接返回)."""
        if self.base_env.done or self.base_env.current_mark == self.learner_mark:
            return
        obs = self._mark_observation(self.base_env.current_mark)
        action = self.opponent(obs, self.config)
        if self.tactical_safety:
            action = safe_policy_action(self.base_env.board, self.base_env.current_mark, self.config, action)
        self.base_env.step(int(action))

    def _terminal_reward(self) -> float:
        """终局奖励 (从学习方视角): 胜 +1 / 负 -1 / 平局或未结束 0."""
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
    """从 MaskablePPO checkpoint 构造推理 agent.

    兼容两种观测: 训练环境的 dict 观测, 或 Kaggle 的原始 board/mark (现场编码)。
    """

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
