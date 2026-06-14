from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from connectx.agents.lookahead import safe_policy_action
from connectx.agents.reward_shaping import RewardShapingConfig, compute_step_reward
from connectx.agents.utils import epsilon_greedy_action
from connectx.envs.connectx_env import (
    ConnectXConfig,
    check_winner,
    encode_board,
    is_draw,
    next_board,
    opponent_mark,
    valid_action_mask,
)


class DQNNet(nn.Module):
    def __init__(self, rows: int = 6, columns: int = 7, channels: int = 64) -> None:
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(channels * rows * columns, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, columns),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class DQNTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    mask: np.ndarray
    next_mask: np.ndarray


class DQNReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.storage: deque[DQNTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: DQNTransition) -> None:
        self.storage.append(transition)

    def sample(self, batch_size: int) -> list[DQNTransition]:
        indices = np.random.choice(len(self.storage), size=batch_size, replace=False)
        return [self.storage[int(idx)] for idx in indices]


class DQNAgent:
    def __init__(
        self,
        config: ConnectXConfig | None = None,
        channels: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.config = config or ConnectXConfig()
        self.gamma = gamma
        self.device = device
        self.online = DQNNet(self.config.rows, self.config.columns, channels).to(device)
        self.target = DQNNet(self.config.rows, self.config.columns, channels).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=learning_rate)

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        self.online.eval()
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(tensor).squeeze(0).detach().cpu().numpy()

    def act(self, board: list[int], mark: int, epsilon: float = 0.0, tactical_safety: bool = False) -> int:
        state = encode_board(board, mark, self.config.rows, self.config.columns)
        mask = valid_action_mask(board, self.config.rows, self.config.columns).astype(bool)
        action = epsilon_greedy_action(self.q_values(state), mask, epsilon)
        if tactical_safety:
            action = safe_policy_action(board, mark, self.config, action)
        return int(action)

    def train_step(self, replay: DQNReplayBuffer, batch_size: int) -> dict[str, float]:
        batch = replay.sample(batch_size)
        states = torch.as_tensor(np.stack([x.state for x in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([x.action for x in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([x.reward for x in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([x.next_state for x in batch]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor([x.done for x in batch], dtype=torch.bool, device=self.device)
        next_masks = torch.as_tensor(np.stack([x.next_mask for x in batch]), dtype=torch.bool, device=self.device)

        self.online.train()
        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).masked_fill(~next_masks, -1e9).max(dim=1).values
            target = rewards - self.gamma * next_q * (~done).float()
        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu())}

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

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
        torch.save(
            {
                "config": self.config.__dict__,
                "gamma": self.gamma,
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DQNAgent":
        payload = torch.load(path, map_location=device)
        agent = cls(config=ConnectXConfig(**payload["config"]), gamma=float(payload["gamma"]), device=device)
        agent.online.load_state_dict(payload["online_state_dict"])
        agent.target.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return agent


def make_dqn_agent(path: str | Path, device: str = "cpu", tactical_safety: bool = True):
    def agent(obs: dict[str, Any], config: ConnectXConfig) -> int:
        if not hasattr(agent, "_dqn"):
            agent._dqn = DQNAgent.load(path, device=device)  # type: ignore[attr-defined]
        dqn = agent._dqn  # type: ignore[attr-defined]
        board = [int(x) for x in obs["board"]]
        mark = int(obs["mark"])
        return dqn.act(board, mark, epsilon=0.0, tactical_safety=tactical_safety)

    return agent


def train_dqn_selfplay(
    episodes: int,
    *,
    config: ConnectXConfig | None = None,
    channels: int = 64,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    replay_capacity: int = 100_000,
    learning_starts: int = 1_000,
    target_sync: int = 500,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    device: str = "cpu",
    reward_shaping: RewardShapingConfig | None = None,
) -> tuple[DQNAgent, list[dict[str, float]]]:
    config = config or ConnectXConfig()
    agent = DQNAgent(config=config, channels=channels, gamma=gamma, learning_rate=learning_rate, device=device)
    replay = DQNReplayBuffer(capacity=replay_capacity)
    curve: list[dict[str, float]] = []
    steps = 0

    for episode in range(episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - episode / max(episodes - 1, 1))
        board = [0] * (config.rows * config.columns)
        mark = 1
        winner = 0
        episode_losses: list[float] = []

        for _move in range(config.rows * config.columns):
            state = encode_board(board, mark, config.rows, config.columns)
            mask = valid_action_mask(board, config.rows, config.columns).astype(bool)
            action = agent.act(board, mark, epsilon=epsilon, tactical_safety=False)
            before = list(board)
            after = next_board(board, action, mark, config.rows, config.columns)
            reward = compute_step_reward(before, after, mark, config, reward_shaping)
            done = False
            if check_winner(after, mark, config.rows, config.columns, config.inarow):
                done = True
                winner = mark
            elif is_draw(after, config.rows, config.columns):
                done = True
                winner = 0

            next_mark = opponent_mark(mark)
            next_state = encode_board(after, next_mark, config.rows, config.columns)
            next_mask = valid_action_mask(after, config.rows, config.columns).astype(bool)
            replay.add(DQNTransition(state, action, reward, next_state, done, mask, next_mask))

            if len(replay) >= max(batch_size, learning_starts):
                metrics = agent.train_step(replay, batch_size)
                episode_losses.append(metrics["loss"])
            steps += 1
            if steps % target_sync == 0:
                agent.sync_target()

            board = after
            if done:
                break
            mark = next_mark

        curve.append(
            {
                "episode": float(episode),
                "epsilon": float(epsilon),
                "winner": float(winner),
                "replay_size": float(len(replay)),
                "mean_loss": float(np.mean(episode_losses)) if episode_losses else 0.0,
            }
        )
        if (episode + 1) % 1000 == 0 or episode + 1 == episodes:
            print(
                f"[dqn] episode {episode + 1}/{episodes} epsilon={epsilon:.3f} replay={len(replay)} "
                f"loss={curve[-1]['mean_loss']:.4f}",
                flush=True,
            )

    return agent, curve
