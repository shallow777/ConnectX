from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from connectx.agents.alphazero.network import load_checkpoint, predict_policy_value_batch
from connectx.envs.connectx_env import ConnectXConfig, encode_board, valid_action_mask


@dataclass
class InferenceServerHandle:
    process: mp.Process
    request_queue: mp.Queue
    response_queues: list[mp.Queue]

    def shutdown(self) -> None:
        try:
            self.request_queue.put(None)
        except Exception:
            pass
        self.process.join(timeout=15)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)


class RemoteEvaluator:
    """Route MCTS network calls to a shared batched GPU inference server."""

    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        config: ConnectXConfig,
    ) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.config = config
        self._seq = 0

    def __call__(self, board: tuple[int, ...], mark: int) -> tuple[np.ndarray, float]:
        encoded = encode_board(board, mark, self.config.rows, self.config.columns)
        mask = valid_action_mask(board, self.config.rows, self.config.columns)
        seq = self._seq
        self._seq += 1
        self.request_queue.put((self.worker_id, seq, encoded, mask))
        while True:
            response = self.response_queue.get()
            rid = response[0]
            if rid != seq:
                continue
            if len(response) == 3:
                _, policy, value = response
                return policy, float(value)
            _, policies, values = response
            return policies[0], float(values[0])

    def evaluate_batch(
        self,
        boards: list[tuple[int, ...]],
        marks: list[int],
    ) -> list[tuple[np.ndarray, float]]:
        if not boards:
            return []
        if len(boards) == 1:
            policy, value = self(boards[0], marks[0])
            return [(policy, value)]

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
        seq = self._seq
        self._seq += 1
        self.request_queue.put((self.worker_id, seq, encoded, masks))
        while True:
            response = self.response_queue.get()
            rid = response[0]
            if rid != seq:
                continue
            _, policies, values = response
            return [(policies[i], float(values[i])) for i in range(len(boards))]


def _run_inference(
    model,
    device: str,
    worker_id: int,
    seq_id: int,
    encoded: np.ndarray,
    masks: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    if encoded.ndim == 3:
        encoded = encoded[None, ...]
        masks = masks[None, ...]
    policies, values = predict_policy_value_batch(model, encoded, masks, device)
    return worker_id, seq_id, policies, values


def _inference_server_loop(
    checkpoint_path: str,
    device: str,
    batch_size: int,
    max_wait_s: float,
    request_queue: mp.Queue,
    response_queues: list[mp.Queue],
) -> None:
    model, _payload = load_checkpoint(checkpoint_path, map_location=device)
    pending: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    def flush(batch: list[tuple[int, int, np.ndarray, np.ndarray]]) -> None:
        if not batch:
            return
        worker_ids = [item[0] for item in batch]
        seq_ids = [item[1] for item in batch]
        encoded = np.stack([item[2] for item in batch], axis=0)
        masks = np.stack([item[3] for item in batch], axis=0)
        policies, values = predict_policy_value_batch(model, encoded, masks, device)
        for worker_id, seq_id, policy, value in zip(worker_ids, seq_ids, policies, values):
            response_queues[worker_id].put((seq_id, policy, float(value)))

    while True:
        try:
            item = request_queue.get(timeout=0.02)
        except queue.Empty:
            if pending:
                flush(pending)
                pending = []
            continue

        if item is None:
            flush(pending)
            break

        worker_id, seq_id, encoded, masks = item
        if encoded.ndim == 4:
            _, _, policies, values = _run_inference(model, device, worker_id, seq_id, encoded, masks)
            response_queues[worker_id].put((seq_id, policies, values))
            continue

        pending.append(item)
        if len(pending) >= batch_size:
            flush(pending)
            pending = []
            continue

        deadline = time.time() + max_wait_s
        while len(pending) < batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                item = request_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                flush(pending)
                return
            worker_id, seq_id, encoded, masks = item
            if encoded.ndim == 4:
                _, _, policies, values = _run_inference(model, device, worker_id, seq_id, encoded, masks)
                response_queues[worker_id].put((seq_id, policies, values))
                continue
            pending.append(item)

        flush(pending)
        pending = []


def start_inference_server(
    checkpoint_path: str | Path,
    *,
    device: str,
    num_workers: int,
    batch_size: int = 64,
    max_wait_ms: float = 2.0,
    mp_context: mp.context.BaseContext | None = None,
) -> InferenceServerHandle:
    ctx = mp_context or mp.get_context("spawn")
    request_queue: mp.Queue = ctx.Queue(maxsize=4096)
    response_queues = [ctx.Queue(maxsize=512) for _ in range(num_workers)]
    process = ctx.Process(
        target=_inference_server_loop,
        args=(
            str(checkpoint_path),
            device,
            batch_size,
            max_wait_ms / 1000.0,
            request_queue,
            response_queues,
        ),
        daemon=True,
    )
    process.start()
    return InferenceServerHandle(process=process, request_queue=request_queue, response_queues=response_queues)
