"""AlphaZero 经验回放池 (replay buffer), 支持 npz 落盘/恢复以便断点续训."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    states: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    masks: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int = 500_000) -> None:
        self.capacity = int(capacity)
        self._states: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._policies: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._values: deque[float] = deque(maxlen=self.capacity)
        self._masks: deque[np.ndarray] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._states)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float, mask: np.ndarray) -> None:
        self._states.append(np.asarray(state, dtype=np.float32))
        self._policies.append(np.asarray(policy, dtype=np.float32))
        self._values.append(float(value))
        self._masks.append(np.asarray(mask, dtype=bool))

    def add_game(self, samples: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]) -> None:
        for state, policy, value, mask in samples:
            self.add(state, policy, value, mask)

    def sample(self, batch_size: int) -> ReplayBatch:
        if len(self) < batch_size:
            raise ValueError(f"Cannot sample batch_size={batch_size} from buffer with {len(self)} samples")
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = np.stack([self._states[idx] for idx in indices]).astype(np.float32)
        policies = np.stack([self._policies[idx] for idx in indices]).astype(np.float32)
        values = np.asarray([self._values[idx] for idx in indices], dtype=np.float32)
        masks = np.stack([self._masks[idx] for idx in indices]).astype(bool)
        return ReplayBatch(states=states, policies=policies, values=values, masks=masks)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=np.asarray(list(self._states), dtype=np.float32),
            policies=np.asarray(list(self._policies), dtype=np.float32),
            values=np.asarray(list(self._values), dtype=np.float32),
            masks=np.asarray(list(self._masks), dtype=bool),
            capacity=np.asarray([self.capacity], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path, capacity: int | None = None) -> "ReplayBuffer":
        data = np.load(path)
        stored_capacity = int(data["capacity"][0]) if "capacity" in data else 500_000
        buffer = cls(capacity=capacity or stored_capacity)
        for state, policy, value, mask in zip(data["states"], data["policies"], data["values"], data["masks"]):
            buffer.add(state, policy, float(value), mask)
        return buffer
