#!/usr/bin/env python3
"""Validate submission/submission_alphazero_rollback_gen8.py acceptance checks."""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kaggle_environments import make

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "submission/versions/submission_alphazero_push_20260611T164545Z.py"
DEFAULT_CANDIDATE = PROJECT_ROOT / "submission/submission_alphazero_rollback_gen8.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs/alphazero_push/checkpoints/generation_0008_accepted.pt"

KAGGLE_CONFIG = {
    "rows": 6,
    "columns": 7,
    "inarow": 4,
    "actTimeout": 2,
    "timeout": 2,
}

KaggleAgent = Callable[[Any, Any], int]


@dataclass
class MatchStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    timeouts: int = 0
    illegal_moves: int = 0
    exceptions: int = 0

    def record_result(self, reward: float) -> None:
        self.games += 1
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1


@dataclass
class ValidationReport:
    head_to_head: MatchStats = field(default_factory=MatchStats)
    candidate_vs_negamax: MatchStats = field(default_factory=MatchStats)
    baseline_vs_negamax: MatchStats = field(default_factory=MatchStats)
    empty_board_step_s: float = 0.0
    midgame_step_s: float = 0.0
    forward_max_error: float = 0.0
    passed: bool = True
    failures: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.passed = False
        self.failures.append(message)


def load_submission_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"submission_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import submission: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_tracked_agent(agent: KaggleAgent, stats: MatchStats) -> KaggleAgent:
    def wrapped(obs: Any, config: Any) -> int:
        try:
            return int(agent(obs, config))
        except Exception:
            stats.exceptions += 1
            raise

    return wrapped


def _scan_episode_issues(env, stats: MatchStats) -> None:
    for step in env.steps:
        for player_state in step:
            status = str(player_state.get("status", ""))
            if status == "TIMEOUT":
                stats.timeouts += 1
            if status.startswith("Invalid"):
                stats.illegal_moves += 1


def run_head_to_head(
    candidate: KaggleAgent,
    baseline: KaggleAgent,
    *,
    games: int,
    configuration: dict[str, Any],
) -> MatchStats:
    stats = MatchStats()
    candidate = make_tracked_agent(candidate, stats)
    baseline = make_tracked_agent(baseline, stats)

    for game_idx in range(games):
        if game_idx % 2 == 0:
            agents = [candidate, baseline]
            candidate_index = 0
        else:
            agents = [baseline, candidate]
            candidate_index = 1

        env = make("connectx", debug=False, configuration=configuration)
        env.reset()
        env.run(agents)
        _scan_episode_issues(env, stats)

        reward = float(env.state[candidate_index].reward)
        stats.record_result(reward)

    return stats


def run_vs_negamax(agent: KaggleAgent, *, games: int, configuration: dict[str, Any]) -> MatchStats:
    stats = MatchStats()
    tracked = make_tracked_agent(agent, stats)
    for _ in range(games):
        env = make("connectx", debug=False, configuration=configuration)
        env.reset()
        env.run([tracked, "negamax"])
        _scan_episode_issues(env, stats)
        stats.record_result(float(env.state[0].reward))
    return stats


def _drop(board: list[int], action: int, mark: int, rows: int, columns: int) -> list[int]:
    new_board = list(board)
    for row in range(rows - 1, -1, -1):
        idx = row * columns + action
        if new_board[idx] == 0:
            new_board[idx] = mark
            return new_board
    return new_board


def _random_midgame_board(rows: int = 6, columns: int = 7, moves: int = 12) -> tuple[list[int], int]:
    board = [0] * (rows * columns)
    mark = 1
    for _ in range(moves):
        legal = [col for col in range(columns) if board[col] == 0]
        if not legal:
            break
        action = random.choice(legal)
        board = _drop(board, action, mark, rows, columns)
        mark = 2 if mark == 1 else 1
    return board, (2 if mark == 1 else 1)


def measure_step_time(module, board: list[int], mark: int, configuration: dict[str, Any]) -> float:
    observation = {"board": board, "mark": mark, "remainingOverageTime": 0.0, "step": 0}
    start = time.perf_counter()
    module.agent(observation, configuration)
    return time.perf_counter() - start


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + residual)


class AlphaZeroTorchNet(nn.Module):
    """Mirror the submission NumPy network for numeric checks."""

    def __init__(self, columns: int = 7, channels: int = 64, residual_blocks: int = 3) -> None:
        super().__init__()
        self.stem = nn.ModuleDict(
            {
                "0": nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
                "1": nn.BatchNorm2d(channels),
            }
        )
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(residual_blocks)])
        self.policy_head = nn.ModuleDict(
            {
                "0": nn.Conv2d(channels, 2, kernel_size=1, bias=False),
                "1": nn.BatchNorm2d(2),
                "4": nn.Linear(2 * 6 * columns, columns),
            }
        )
        self.value_head = nn.ModuleDict(
            {
                "0": nn.Conv2d(channels, 1, kernel_size=1, bias=False),
                "1": nn.BatchNorm2d(1),
                "4": nn.Linear(1 * 6 * columns, channels),
                "6": nn.Linear(channels, 1),
            }
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.stem["1"](self.stem["0"](x)))
        for block in self.blocks:
            x = block(x)
        policy = F.relu(self.policy_head["1"](self.policy_head["0"](x))).reshape(x.size(0), -1)
        logits = self.policy_head["4"](policy)
        value = F.relu(self.value_head["1"](self.value_head["0"](x))).reshape(x.size(0), -1)
        value = F.relu(self.value_head["4"](value))
        value = torch.tanh(self.value_head["6"](value))
        return logits, value.squeeze(-1)


def torch_forward(
    model: AlphaZeroTorchNet,
    board: list[int],
    mark: int,
    *,
    rows: int = 6,
    columns: int = 7,
) -> tuple[np.ndarray, float]:
    arr = np.asarray(board, dtype=np.int8).reshape(rows, columns)
    other = 2 if mark == 1 else 1
    planes = np.stack([(arr == mark).astype(np.float32), (arr == other).astype(np.float32)], axis=0)
    tensor = torch.from_numpy(planes).unsqueeze(0)
    with torch.no_grad():
        logits, value = model(tensor)
    return logits.squeeze(0).cpu().numpy().astype(np.float32), float(value.item())


def check_forward_consistency(
    module,
    checkpoint_path: Path,
    *,
    samples: int = 5,
    seed: int = 0,
) -> float:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network_config = payload["network_config"]
    model = AlphaZeroTorchNet(
        columns=int(network_config["columns"]),
        channels=int(network_config["channels"]),
        residual_blocks=int(network_config["residual_blocks"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    rng = random.Random(seed)
    rows = int(network_config["rows"])
    columns = int(network_config["columns"])
    max_error = 0.0
    for _ in range(samples):
        moves = rng.randint(0, 20)
        board, mark = _random_midgame_board(rows=rows, columns=columns, moves=moves)
        np_logits, np_value = module._forward(board, mark, rows, columns)
        torch_logits, torch_value = torch_forward(model, board, mark, rows=rows, columns=columns)
        max_error = max(
            max_error,
            float(np.max(np.abs(np_logits - torch_logits))),
            abs(np_value - torch_value),
        )
    return max_error


def validate(
    *,
    candidate_path: Path,
    baseline_path: Path,
    checkpoint_path: Path,
    games: int,
) -> ValidationReport:
    report = ValidationReport()
    candidate_mod = load_submission_module(candidate_path)
    baseline_mod = load_submission_module(baseline_path)

    report.head_to_head = run_head_to_head(
        candidate_mod.agent,
        baseline_mod.agent,
        games=games,
        configuration=KAGGLE_CONFIG,
    )
    if report.head_to_head.exceptions:
        report.fail(f"head-to-head exceptions: {report.head_to_head.exceptions}")
    if report.head_to_head.timeouts:
        report.fail(f"head-to-head timeouts: {report.head_to_head.timeouts}")
    if report.head_to_head.illegal_moves:
        report.fail(f"head-to-head illegal moves: {report.head_to_head.illegal_moves}")

    report.baseline_vs_negamax = run_vs_negamax(
        baseline_mod.agent,
        games=games,
        configuration=KAGGLE_CONFIG,
    )
    report.candidate_vs_negamax = run_vs_negamax(
        candidate_mod.agent,
        games=games,
        configuration=KAGGLE_CONFIG,
    )
    if report.candidate_vs_negamax.exceptions:
        report.fail(f"candidate vs negamax exceptions: {report.candidate_vs_negamax.exceptions}")
    if report.candidate_vs_negamax.timeouts:
        report.fail(f"candidate vs negamax timeouts: {report.candidate_vs_negamax.timeouts}")
    if report.candidate_vs_negamax.illegal_moves:
        report.fail(f"candidate vs negamax illegal moves: {report.candidate_vs_negamax.illegal_moves}")

    baseline_rate = report.baseline_vs_negamax.wins / max(1, report.baseline_vs_negamax.games)
    candidate_rate = report.candidate_vs_negamax.wins / max(1, report.candidate_vs_negamax.games)
    if candidate_rate + 1e-9 < baseline_rate:
        report.fail(
            "candidate negamax win rate "
            f"{candidate_rate:.3f} < baseline {baseline_rate:.3f}"
        )

    report.empty_board_step_s = measure_step_time(
        candidate_mod,
        [0] * 42,
        1,
        KAGGLE_CONFIG,
    )
    mid_board, mid_mark = _random_midgame_board(moves=12)
    report.midgame_step_s = measure_step_time(candidate_mod, mid_board, mid_mark, KAGGLE_CONFIG)
    if report.empty_board_step_s >= 1.9:
        report.fail(f"empty-board step {report.empty_board_step_s:.3f}s >= 1.9s")
    if report.midgame_step_s >= 1.9:
        report.fail(f"midgame step {report.midgame_step_s:.3f}s >= 1.9s")

    report.forward_max_error = check_forward_consistency(
        candidate_mod,
        checkpoint_path,
        samples=5,
        seed=0,
    )
    if report.forward_max_error >= 1e-4:
        report.fail(f"forward max error {report.forward_max_error:.6e} >= 1e-4")

    return report


def print_report(report: ValidationReport) -> None:
    h2h = report.head_to_head
    print("=== ConnectX submission validation ===")
    print(
        f"1) head-to-head ({h2h.games} games): "
        f"W {h2h.wins} / L {h2h.losses} / D {h2h.draws} | "
        f"timeouts={h2h.timeouts} illegal={h2h.illegal_moves} exceptions={h2h.exceptions}"
    )
    base = report.baseline_vs_negamax
    cand = report.candidate_vs_negamax
    print(
        f"2) vs negamax baseline ({base.games} games): "
        f"win_rate={base.wins / max(1, base.games):.3f} "
        f"(W {base.wins} L {base.losses} D {base.draws})"
    )
    print(
        f"   vs negamax candidate ({cand.games} games): "
        f"win_rate={cand.wins / max(1, cand.games):.3f} "
        f"(W {cand.wins} L {cand.losses} D {cand.draws})"
    )
    print(
        f"3) step timing: empty={report.empty_board_step_s:.3f}s "
        f"midgame={report.midgame_step_s:.3f}s (limit 1.9s)"
    )
    print(f"4) forward max error vs torch checkpoint: {report.forward_max_error:.6e} (limit 1e-4)")
    if report.passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        for item in report.failures:
            print(f" - {item}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--games", type=int, default=20)
    args = parser.parse_args(argv)

    report = validate(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        checkpoint_path=args.checkpoint,
        games=args.games,
    )
    print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
