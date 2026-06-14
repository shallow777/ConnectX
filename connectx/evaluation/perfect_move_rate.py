#!/usr/bin/env python3
"""Perfect-move-rate evaluation using the bitboard exact solver."""

from __future__ import annotations

import argparse
import sys
import importlib.util
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectx.solver.bitboard import BitboardSolver, classify_phase, count_pieces
DEFAULT_REPLAY = PROJECT_ROOT / "runs/alphazero_push/replay_buffer.npz"
DEFAULT_BASELINE = PROJECT_ROOT / "submission/versions/submission_alphazero_push_20260611T164545Z.py"
DEFAULT_CANDIDATE = PROJECT_ROOT / "submission/submission_alphazero_rollback_gen8.py"

def make_config(act_timeout: float) -> dict:
    return {
        "rows": 6,
        "columns": 7,
        "inarow": 4,
        "actTimeout": act_timeout,
        "timeout": act_timeout,
    }


@dataclass
class PhaseStats:
    total: int = 0
    perfect: int = 0

    def rate(self) -> float:
        return self.perfect / self.total if self.total else 0.0


@dataclass
class AgentReport:
    name: str
    overall: PhaseStats = field(default_factory=PhaseStats)
    opening: PhaseStats = field(default_factory=PhaseStats)
    midgame: PhaseStats = field(default_factory=PhaseStats)
    endgame: PhaseStats = field(default_factory=PhaseStats)

    def record(self, phase: str, is_perfect: bool) -> None:
        self.overall.total += 1
        self.overall.perfect += int(is_perfect)
        bucket = getattr(self, phase)
        bucket.total += 1
        bucket.perfect += int(is_perfect)


@dataclass
class ComparisonReport:
    positions: int
    baseline: AgentReport
    candidate: AgentReport

    def to_dict(self) -> dict:
        def pack(stats: AgentReport) -> dict:
            return {
                "overall": {"total": stats.overall.total, "perfect": stats.overall.perfect, "rate": stats.overall.rate()},
                "opening": {"total": stats.opening.total, "perfect": stats.opening.perfect, "rate": stats.opening.rate()},
                "midgame": {"total": stats.midgame.total, "perfect": stats.midgame.perfect, "rate": stats.midgame.rate()},
                "endgame": {"total": stats.endgame.total, "perfect": stats.endgame.perfect, "rate": stats.endgame.rate()},
            }

        return {
            "positions": self.positions,
            "baseline": pack(self.baseline),
            "candidate": pack(self.candidate),
        }


def load_submission_agent(path: Path):
    spec = importlib.util.spec_from_file_location(f"submission_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import submission: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def state_planes_to_board_mark(planes: np.ndarray, *, mark: int = 1) -> tuple[list[int], int]:
    rows, columns = planes.shape[1], planes.shape[2]
    board = [0] * (rows * columns)
    other = 2 if mark == 1 else 1
    for row in range(rows):
        for col in range(columns):
            if planes[0, row, col] > 0.5:
                board[row * columns + col] = mark
            elif planes[1, row, col] > 0.5:
                board[row * columns + col] = other
    return board, mark


def load_replay_positions(
    replay_path: Path,
    *,
    min_midgame: int,
    max_positions: int,
    seed: int,
) -> list[tuple[list[int], int, str]]:
    data = np.load(replay_path, allow_pickle=False)
    states = data["states"]
    rng = random.Random(seed)

    indexed: list[tuple[int, str]] = []
    for idx in range(states.shape[0]):
        board, _ = state_planes_to_board_mark(states[idx])
        phase = classify_phase(board)
        indexed.append((idx, phase))

    midgame_indices = [idx for idx, phase in indexed if phase == "midgame"]
    if len(midgame_indices) < min_midgame:
        raise RuntimeError(
            f"Replay buffer only has {len(midgame_indices)} midgame positions; need {min_midgame}"
        )

    rng.shuffle(midgame_indices)
    chosen_mid = midgame_indices[:min_midgame]
    opening = [idx for idx, phase in indexed if phase == "opening"]
    endgame = [idx for idx, phase in indexed if phase == "endgame"]
    rng.shuffle(opening)
    rng.shuffle(endgame)

    extra = max(0, max_positions - len(chosen_mid))
    opening_take = min(len(opening), extra // 2)
    endgame_take = min(len(endgame), extra - opening_take)
    selected = chosen_mid + opening[:opening_take] + endgame[:endgame_take]
    rng.shuffle(selected)

    positions: list[tuple[list[int], int, str]] = []
    for idx in selected:
        board, mark = state_planes_to_board_mark(states[idx])
        positions.append((board, mark, classify_phase(board)))
    return positions


def annotate_optimal_moves(
    positions: list[tuple[list[int], int, str]],
    solver: BitboardSolver,
) -> list[tuple[list[int], int, str, tuple[int, ...]]]:
    annotated: list[tuple[list[int], int, str, tuple[int, ...]]] = []
    total = len(positions)
    for index, (board, mark, phase) in enumerate(positions, start=1):
        optimal = solver.optimal_moves(board, mark)
        if optimal:
            annotated.append((board, mark, phase, optimal))
        if index % 100 == 0 or index == total:
            print(f"[solver] annotated {index}/{total}", flush=True)
    return annotated


def evaluate_agent_on_positions(
    name: str,
    agent,
    positions: list[tuple[list[int], int, str, tuple[int, ...]]],
    *,
    config: dict,
) -> AgentReport:
    report = AgentReport(name=name)
    total = len(positions)
    for index, (board, mark, phase, optimal) in enumerate(positions, start=1):
        if not optimal:
            continue
        obs = {"board": board, "mark": mark, "remainingOverageTime": 0.0, "step": count_pieces(board)}
        chosen = int(agent(obs, config))
        report.record(phase, chosen in optimal)
        if index % 25 == 0 or index == total:
            print(f"[{name}] {index}/{total}", flush=True)
    return report


def run_comparison(
    *,
    baseline_path: Path,
    candidate_path: Path,
    replay_path: Path,
    min_midgame: int,
    max_positions: int,
    seed: int,
    act_timeout: float,
) -> ComparisonReport:
    positions = load_replay_positions(
        replay_path,
        min_midgame=min_midgame,
        max_positions=max_positions,
        seed=seed,
    )
    solver = BitboardSolver()
    annotated = annotate_optimal_moves(positions, solver)
    baseline_agent = load_submission_agent(baseline_path)
    candidate_agent = load_submission_agent(candidate_path)

    config = make_config(act_timeout)
    baseline_report = evaluate_agent_on_positions(
        "baseline_682", baseline_agent, annotated, config=config
    )
    candidate_report = evaluate_agent_on_positions(
        "candidate_gen8", candidate_agent, annotated, config=config
    )
    return ComparisonReport(positions=len(annotated), baseline=baseline_report, candidate=candidate_report)


def print_report(report: ComparisonReport) -> None:
    print("=== Perfect Move Rate Report ===")
    print(f"positions evaluated: {report.positions}")
    for label, stats in (("682 baseline (gen4)", report.baseline), ("gen8 rollback", report.candidate)):
        print(f"\n{label}:")
        for phase in ("overall", "opening", "midgame", "endgame"):
            bucket = getattr(stats, phase)
            print(f"  {phase:8s}: {bucket.perfect:4d}/{bucket.total:4d} = {bucket.rate():.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--min-midgame", type=int, default=500)
    parser.add_argument("--max-positions", type=int, default=700)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--act-timeout", type=float, default=2.0)
    args = parser.parse_args(argv)

    report = run_comparison(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        replay_path=args.replay,
        min_midgame=args.min_midgame,
        max_positions=args.max_positions,
        seed=args.seed,
        act_timeout=args.act_timeout,
    )
    print_report(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
