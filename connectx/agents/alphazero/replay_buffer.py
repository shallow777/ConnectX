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
    def __init__(self, capacity: int = 500_000, *, max_generations: int | None = None) -> None:
        self.capacity = int(capacity)
        self.max_generations = max_generations
        self._states: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._policies: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._values: deque[float] = deque(maxlen=self.capacity)
        self._masks: deque[np.ndarray] = deque(maxlen=self.capacity)
        self._generations: deque[int] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._states)

    def add(
        self,
        state: np.ndarray,
        policy: np.ndarray,
        value: float,
        mask: np.ndarray,
        *,
        generation: int = 0,
    ) -> None:
        self._states.append(np.asarray(state, dtype=np.float32))
        self._policies.append(np.asarray(policy, dtype=np.float32))
        self._values.append(float(value))
        self._masks.append(np.asarray(mask, dtype=bool))
        self._generations.append(int(generation))

    def add_game(
        self,
        samples: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]],
        *,
        generation: int = 0,
    ) -> None:
        for state, policy, value, mask in samples:
            self.add(state, policy, value, mask, generation=generation)
        self._prune_by_generation()

    def _prune_by_generation(self) -> None:
        if self.max_generations is None or len(self) == 0:
            return
        newest = max(self._generations)
        cutoff = newest - self.max_generations + 1
        if min(self._generations) >= cutoff:
            return
        kept = [
            (state, policy, value, mask, gen)
            for state, policy, value, mask, gen in zip(
                self._states,
                self._policies,
                self._values,
                self._masks,
                self._generations,
            )
            if gen >= cutoff
        ]
        self._states.clear()
        self._policies.clear()
        self._values.clear()
        self._masks.clear()
        self._generations.clear()
        for state, policy, value, mask, gen in kept:
            self.add(state, policy, value, mask, generation=gen)

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
        payload: dict[str, np.ndarray] = {
            "states": np.asarray(list(self._states), dtype=np.float32),
            "policies": np.asarray(list(self._policies), dtype=np.float32),
            "values": np.asarray(list(self._values), dtype=np.float32),
            "masks": np.asarray(list(self._masks), dtype=bool),
            "capacity": np.asarray([self.capacity], dtype=np.int64),
        }
        if self._generations:
            payload["generations"] = np.asarray(list(self._generations), dtype=np.int64)
        if self.max_generations is not None:
            payload["max_generations"] = np.asarray([self.max_generations], dtype=np.int64)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(
        cls,
        path: str | Path,
        capacity: int | None = None,
        max_generations: int | None = None,
    ) -> "ReplayBuffer":
        data = np.load(path)
        stored_capacity = int(data["capacity"][0]) if "capacity" in data else 500_000
        stored_max_generations = (
            int(data["max_generations"][0]) if "max_generations" in data else max_generations
        )
        generations = data["generations"] if "generations" in data else [0] * len(data["states"])
        buffer = cls(
            capacity=capacity or stored_capacity,
            max_generations=max_generations if max_generations is not None else stored_max_generations,
        )
        for state, policy, value, mask, generation in zip(
            data["states"],
            data["policies"],
            data["values"],
            data["masks"],
            generations,
        ):
            buffer.add(state, policy, float(value), mask, generation=int(generation))
        return buffer
