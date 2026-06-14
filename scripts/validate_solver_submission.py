#!/usr/bin/env python3
"""Validate solver-enhanced submission against rollback_gen8 baseline."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kaggle_environments import make
DEFAULT_BASELINE = PROJECT_ROOT / "submission/submission_alphazero_rollback_gen8.py"
DEFAULT_CANDIDATE = PROJECT_ROOT / "submission/submission_alphazero_solver_gen8.py"

KAGGLE_CONFIG = {
    "rows": 6,
    "columns": 7,
    "inarow": 4,
    "actTimeout": 2,
    "timeout": 2,
}


@dataclass
class MatchStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    timeouts: int = 0
    illegal_moves: int = 0
    exceptions: int = 0

    def record(self, reward: float) -> None:
        self.games += 1
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1

    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


@dataclass
class ValidationReport:
    head_to_head: MatchStats = field(default_factory=MatchStats)
    endgame_passed: int = 0
    endgame_total: int = 0
    pmr_candidate: float = 0.0
    pmr_baseline: float = 0.0
    passed: bool = True
    failures: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.passed = False
        self.failures.append(message)


def load_agent(path: Path):
    spec = importlib.util.spec_from_file_location(f"submission_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def _scan_issues(env, stats: MatchStats) -> None:
    for step in env.steps:
        for player_state in step:
            status = str(player_state.get("status", ""))
            if status == "TIMEOUT":
                stats.timeouts += 1
            if status.startswith("Invalid"):
                stats.illegal_moves += 1


def run_head_to_head(candidate, baseline, *, games: int) -> MatchStats:
    stats = MatchStats()
    for game_idx in range(games):
        if game_idx % 2 == 0:
            agents = [candidate, baseline]
            candidate_index = 0
        else:
            agents = [baseline, candidate]
            candidate_index = 1
        env = make("connectx", debug=False, configuration=KAGGLE_CONFIG)
        env.reset()
        try:
            env.run(agents)
        except Exception:
            stats.exceptions += 1
            continue
        _scan_issues(env, stats)
        stats.record(float(env.state[candidate_index].reward))
    return stats


def _drop_piece(board: list[int], col: int, mark: int) -> list[int]:
    board = list(board)
    for row in range(5, -1, -1):
        idx = row * 7 + col
        if board[idx] == 0:
            board[idx] = mark
            return board
    return board


def _planes_to_board_mark(planes) -> tuple[list[int], int]:
    board = [0] * 42
    mark = 1
    for row in range(6):
        for col in range(7):
            if planes[0, row, col] > 0.5:
                board[row * 7 + col] = mark
            elif planes[1, row, col] > 0.5:
                board[row * 7 + col] = 2 if mark == 1 else 1
    return board, mark


def build_winning_endgames() -> list[tuple[list[int], int, int]]:
    """Ten forced-win positions with <=16 empty cells."""
    import numpy as np

    from connectx.solver.bitboard import BitboardSolver, count_empty

    solver = BitboardSolver()
    puzzles: list[tuple[list[int], int, int]] = []
    seen: set[str] = set()

    data = np.load(PROJECT_ROOT / "runs/alphazero_push/replay_buffer.npz", allow_pickle=False)
    for planes in data["states"]:
        board, mark = _planes_to_board_mark(planes)
        if count_empty(board) > 16:
            continue
        scores = solver.move_scores(board, mark)
        if not scores or max(scores.values()) != 1:
            continue
        optimal = solver.optimal_moves(board, mark)
        if not optimal:
            continue
        key = "".join(map(str, board)) + str(mark)
        if key in seen:
            continue
        seen.add(key)
        puzzles.append((board, mark, optimal[0]))
        if len(puzzles) >= 10:
            break

    if len(puzzles) < 10:
        raise RuntimeError(f"only built {len(puzzles)} winning endgame puzzles")
    return puzzles


def validate_endgames(agent, puzzles) -> tuple[int, int]:
    passed = 0
    for board, mark, expected in puzzles:
        obs = {"board": board, "mark": mark, "remainingOverageTime": 0.0, "step": 0}
        action = int(agent(obs, KAGGLE_CONFIG))
        if action == expected:
            passed += 1
    return passed, len(puzzles)


def validate(
    *,
    candidate_path: Path,
    baseline_path: Path,
    games: int,
    skip_pmr: bool,
) -> ValidationReport:
    report = ValidationReport()
    candidate = load_agent(candidate_path)
    baseline = load_agent(baseline_path)

    report.head_to_head = run_head_to_head(candidate, baseline, games=games)
    if report.head_to_head.exceptions:
        report.fail(f"exceptions: {report.head_to_head.exceptions}")
    if report.head_to_head.timeouts:
        report.fail(f"timeouts: {report.head_to_head.timeouts}")
    if report.head_to_head.illegal_moves:
        report.fail(f"illegal moves: {report.head_to_head.illegal_moves}")
    if report.head_to_head.win_rate() < 0.55:
        report.fail(
            f"win rate {report.head_to_head.win_rate():.3f} < 0.55 "
            f"(W{report.head_to_head.wins}/L{report.head_to_head.losses}/D{report.head_to_head.draws})"
        )

    puzzles = build_winning_endgames()
    report.endgame_passed, report.endgame_total = validate_endgames(candidate, puzzles)
    if report.endgame_passed != report.endgame_total:
        report.fail(f"endgame puzzles {report.endgame_passed}/{report.endgame_total}")

    if not skip_pmr:
        from connectx.evaluation.perfect_move_rate import run_comparison

        pmr = run_comparison(
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            replay_path=PROJECT_ROOT / "runs/alphazero_push/replay_buffer.npz",
            min_midgame=200,
            max_positions=300,
            seed=0,
            act_timeout=2.0,
        )
        report.pmr_baseline = pmr.baseline.overall.rate()
        report.pmr_candidate = pmr.candidate.overall.rate()
        if report.pmr_candidate + 1e-9 < report.pmr_baseline:
            report.fail(
                f"perfect move rate {report.pmr_candidate:.3f} < baseline {report.pmr_baseline:.3f}"
            )

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--skip-pmr", action="store_true")
    args = parser.parse_args(argv)

    report = validate(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        games=args.games,
        skip_pmr=args.skip_pmr,
    )

    h = report.head_to_head
    print("=== Solver submission validation ===")
    print(
        f"1) vs baseline ({h.games} games): W{h.wins} L{h.losses} D{h.draws} "
        f"rate={h.win_rate():.3f} timeouts={h.timeouts} illegal={h.illegal_moves}"
    )
    print(f"2) endgame puzzles: {report.endgame_passed}/{report.endgame_total}")
    if not args.skip_pmr:
        print(
            f"3) perfect move rate: candidate={report.pmr_candidate:.3f} "
            f"baseline={report.pmr_baseline:.3f}"
        )
    print("RESULT:", "PASS" if report.passed else "FAIL")
    for item in report.failures:
        print(" -", item)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
