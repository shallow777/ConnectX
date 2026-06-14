#!/usr/bin/env python3
"""Validate submission/submission_alphazero_hybrid_gen8.py vs cached baseline."""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_HYBRID = PROJECT_ROOT / "submission/submission_alphazero_hybrid_gen8.py"
DEFAULT_CACHED = PROJECT_ROOT / "submission/submission_alphazero_gen8_cached.py"


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    gates: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.gates.append(GateResult(name, ok, detail))


def load_agent(path: Path):
    spec = importlib.util.spec_from_file_location(f"submission_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _drop_naive(board, action, mark, rows, columns):
    new = list(board)
    for row in range(rows - 1, -1, -1):
        idx = row * columns + action
        if new[idx] == 0:
            new[idx] = mark
            return new
    return new


def _winner_naive(board, mark, rows, columns, inarow):
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(rows):
        for col in range(columns):
            if board[row * columns + col] != mark:
                continue
            for dr, dc in directions:
                end_row = row + (inarow - 1) * dr
                end_col = col + (inarow - 1) * dc
                if end_row < 0 or end_row >= rows or end_col < 0 or end_col >= columns:
                    continue
                if all(
                    board[(row + offset * dr) * columns + col + offset * dc] == mark
                    for offset in range(inarow)
                ):
                    return True
    return False


def _random_positions(count: int, seed: int) -> list[tuple[list[int], int]]:
    rng = random.Random(seed)
    positions: list[tuple[list[int], int]] = []
    rows, columns = 6, 7
    while len(positions) < count:
        board = [0] * (rows * columns)
        mark = 1
        for _ in range(rng.randint(0, 20)):
            legal = [c for c in range(columns) if board[c] == 0]
            if not legal:
                break
            col = rng.choice(legal)
            board = _drop_naive(board, col, mark, rows, columns)
            positions.append((list(board), mark))
            mark = 2 if mark == 1 else 1
    return positions


def gate_a_bitboard_parity(module, samples: int, seed: int) -> GateResult:
    rows, columns, inarow = 6, 7, 4
    rng = random.Random(seed)
    for board, mark in _random_positions(samples, seed):
        legal = [col for col in range(columns) if board[col] == 0]
        if not legal:
            continue
        for col in legal:
            a = _drop_naive(board, col, mark, rows, columns)
            b = module._drop(board, col, mark, rows, columns)
            if a != b:
                return GateResult("A3 bitboard parity", False, f"drop mismatch col={col}")
            for check_mark in (1, 2):
                if _winner_naive(a, check_mark, rows, columns, inarow) != module._winner(
                    a, check_mark, rows, columns, inarow
                ):
                    return GateResult("A3 bitboard parity", False, f"winner mismatch mark={check_mark}")
        probe = rng.choice(legal)
        after = _drop_naive(board, probe, mark, rows, columns)
        for check_mark in (1, 2):
            if _winner_naive(after, check_mark, rows, columns, inarow) != module._winner(
                after, check_mark, rows, columns, inarow
            ):
                return GateResult("A3 bitboard parity", False, f"winner probe mark={check_mark}")
    return GateResult("A3 bitboard parity", True, f"{samples} positions zero diff")


def gate_a_opening(agent) -> GateResult:
    config = {"rows": 6, "columns": 7, "inarow": 4, "actTimeout": 2.0, "timeout": 2.0}
    obs = {"board": [0] * 42, "mark": 1, "remainingOverageTime": 0.0, "step": 0}
    move = int(agent(obs, config))
    ok = move == 3
    return GateResult("A4 opening center", ok, f"first move={move}")


def _scan_issues(env) -> tuple[int, int]:
    timeouts = illegal = 0
    for step in env.steps:
        for player_state in step:
            status = str(player_state.get("status", ""))
            if status == "TIMEOUT":
                timeouts += 1
            if status.startswith("Invalid"):
                illegal += 1
    return timeouts, illegal


def play_match(agent_a, agent_b, config: dict, first_a: bool) -> tuple[float, int, int]:
    from kaggle_environments import make

    agents = [agent_a, agent_b] if first_a else [agent_b, agent_a]
    idx = 0 if first_a else 1
    env = make("connectx", debug=False, configuration=config)
    env.reset()
    env.run(agents)
    to, il = _scan_issues(env)
    return float(env.state[idx].reward), to, il


def gate_a_safety(agent, opponent, config: dict, games: int) -> GateResult:
    timeouts = illegal = exceptions = 0
    for i in range(games):
        first = i % 2 == 0
        try:
            _, to, il = play_match(agent, opponent, config, first)
        except Exception:
            exceptions += 1
            continue
        timeouts += to
        illegal += il
    ok = timeouts == 0 and illegal == 0 and exceptions == 0
    detail = f"timeouts={timeouts} illegal={illegal} exceptions={exceptions} over {games} games"
    return GateResult("A1 safety", ok, detail)


def build_winning_endgames() -> list[tuple[list[int], int, list[int]]]:
    from connectx.solver.bitboard import BitboardSolver, count_empty

    solver = BitboardSolver()
    puzzles: list[tuple[list[int], int, list[int]]] = []
    data = np.load(PROJECT_ROOT / "runs/alphazero_push/replay_buffer.npz")
    for planes in data["states"]:
        board = [0] * 42
        mark = 1
        for row in range(6):
            for col in range(7):
                if planes[0, row, col] > 0.5:
                    board[row * 7 + col] = mark
                elif planes[1, row, col] > 0.5:
                    board[row * 7 + col] = 2 if mark == 1 else 1
        if count_empty(board) > 16:
            continue
        scores = solver.move_scores(board, mark)
        if scores and max(scores.values()) == 1:
            opt = solver.optimal_moves(board, mark)
            puzzles.append((board, mark, opt))
        if len(puzzles) >= 10:
            break
    return puzzles


def gate_a_endgames(agent, config: dict) -> GateResult:
    puzzles = build_winning_endgames()
    passed = 0
    for board, mark, optimal in puzzles:
        obs = {"board": board, "mark": mark, "remainingOverageTime": 0.0, "step": 0}
        move = int(agent(obs, config))
        if move in optimal:
            passed += 1
    ok = passed == len(puzzles) == 10
    return GateResult("A2 endgame wins", ok, f"{passed}/{len(puzzles)} in optimal set")


def _drop_naive(board, action, mark, rows=6, columns=7):
    new = list(board)
    for row in range(rows - 1, -1, -1):
        idx = row * columns + action
        if new[idx] == 0:
            new[idx] = mark
            return new
    return new


def build_losing_fight_positions() -> list[tuple[list[int], int]]:
    """Theoretically lost positions (solver root -1) where hybrid does not filter root moves."""
    import importlib.util

    from connectx.solver.bitboard import BitboardSolver, count_empty

    hybrid_path = PROJECT_ROOT / "submission/submission_alphazero_hybrid_gen8.py"
    spec = importlib.util.spec_from_file_location("hybrid_probe", hybrid_path)
    hybrid_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_mod)
    hybrid_mod._SOLVER_MAX_EMPTY = 18

    solver = BitboardSolver()
    out: list[tuple[list[int], int]] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    rng = random.Random(12)

    for _trial in range(50000):
        board = [0] * 42
        mark = 1
        for _ in range(rng.randint(2, 12)):
            legal = [col for col in range(7) if board[col] == 0]
            if not legal:
                break
            board = _drop_naive(board, rng.choice(legal), mark)
            mark = 2 if mark == 1 else 1
        if count_empty(board) > 40:
            continue
        for side in (1, 2):
            result = solver.solve_unlimited(board, side)
            if not (result.completed and result.score == -1):
                continue
            mode, payload = hybrid_mod._bb_classify_root(board, side, time.time() + 15.0)
            if mode != "filter" or payload is not None:
                continue
            key = (tuple(board), side)
            if key in seen:
                continue
            seen.add(key)
            out.append((list(board), side))
        if len(out) >= 10:
            break
    return out[:10]


def gate_a_losing_fallback(hybrid_mod, cached_agent, config: dict) -> GateResult:
    positions = build_losing_fight_positions()
    if len(positions) < 10:
        return GateResult("A2b losing fight", False, f"only {len(positions)} positions")
    ok_count = 0
    hybrid_mod._SOLVER_MAX_EMPTY = 18
    for board, mark in positions:
        mode, payload = hybrid_mod._bb_classify_root(board, mark, time.time() + 15.0)
        if mode != "filter" or payload is not None:
            continue
        obs = {"board": board, "mark": mark, "remainingOverageTime": 0.0, "step": 0}
        hybrid_mod._SOLVER_MAX_EMPTY = 18
        hybrid_move = int(hybrid_mod.agent(obs, config))
        cached_move = int(cached_agent(dict(obs), config))
        if hybrid_move == cached_move:
            ok_count += 1
    ok = ok_count >= 10
    return GateResult("A2b losing fight", ok, f"{ok_count}/{len(positions)} match cached")


def make_opening_board(moves: list[int]) -> tuple[list[int], int]:
    board = [0] * 42
    mark = 1
    for col in moves:
        board = _drop_naive(board, col, mark, 6, 7)
        mark = 2 if mark == 1 else 1
    return board, mark


def _play_from_opening(hybrid_agent, cached_agent, board, mark, *, first_hybrid, config) -> float:
    hybrid_mark = mark if first_hybrid else (2 if mark == 1 else 1)
    current_board = list(board)
    current_mark = mark
    reward = 0.0
    for _ in range(42 - sum(1 for cell in board if cell)):
        agent = hybrid_agent if current_mark == hybrid_mark else cached_agent
        obs = {"board": current_board, "mark": current_mark, "remainingOverageTime": 0.0, "step": 0}
        action = int(agent(obs, config))
        if current_board[action] != 0:
            break
        current_board = _drop_naive(current_board, action, current_mark, 6, 7)
        if _winner_naive(current_board, current_mark, 6, 7, 4):
            reward = 1.0 if current_mark == hybrid_mark else -1.0
            break
        if all(current_board[col] != 0 for col in range(7)):
            reward = 0.0
            break
        current_mark = 2 if current_mark == 1 else 1
    if reward > 0:
        return 1.0
    if reward < 0:
        return 0.0
    return 0.5


def gate_b_fast_arena(hybrid_agent, cached_agent) -> GateResult:
    openings = [
        [],
        [3],
        [3, 3],
        [3, 4],
        [3, 2],
        [3, 3, 4],
        [3, 4, 3],
        [3, 3, 2],
        [3, 2, 4],
        [3, 4, 2],
        [3, 3, 4, 4],
        [3, 4, 3, 3],
        [3, 2, 3, 2],
        [3, 3, 2, 2],
        [3, 4, 2, 3],
        [3, 2, 4, 3],
        [3, 3, 4, 2],
        [3, 4, 3, 2],
        [3, 2, 3, 4],
        [3, 3, 3],
    ]
    config = {"rows": 6, "columns": 7, "inarow": 4, "actTimeout": 0.4, "timeout": 0.4}
    score = 0.0
    games = 0
    for opening in openings:
        board, mark = make_opening_board(opening)
        for first_hybrid in (True, False):
            for _round in range(5):
                score += _play_from_opening(
                    hybrid_agent, cached_agent, board, mark, first_hybrid=first_hybrid, config=config
                )
                games += 1
    rate = score / games if games else 0.0
    return GateResult("B1 fast arena", rate >= 0.50, f"score {score}/{games} = {rate:.3f}")


def gate_b_full_time(hybrid_agent, cached_agent, games: int = 20) -> GateResult:
    config = {"rows": 6, "columns": 7, "inarow": 4, "actTimeout": 2.0, "timeout": 2.0}
    score = 0.0
    timeouts = illegal = 0
    for i in range(games):
        first_hybrid = i % 2 == 0
        reward, to, il = play_match(hybrid_agent, cached_agent, config, first_hybrid)
        timeouts += to
        illegal += il
        if reward > 0:
            score += 1.0
        elif reward == 0:
            score += 0.5
    rate = score / games
    ok = rate >= 0.40 and timeouts == 0 and illegal == 0
    return GateResult(
        "B2 full-time arena",
        ok,
        f"score {score}/{games}={rate:.3f} timeouts={timeouts} illegal={illegal}",
    )


def gate_c_pmr(hybrid_path: Path, cached_path: Path) -> GateResult:
    from connectx.evaluation.perfect_move_rate import run_comparison

    report = run_comparison(
        baseline_path=cached_path,
        candidate_path=hybrid_path,
        replay_path=PROJECT_ROOT / "runs/alphazero_push/replay_buffer.npz",
        min_midgame=200,
        max_positions=300,
        seed=0,
        act_timeout=2.0,
    )
    end_ok = report.candidate.endgame.rate() + 1e-9 >= report.baseline.endgame.rate()
    global_ok = report.candidate.overall.rate() + 0.01 >= report.baseline.overall.rate()
    detail = (
        f"hybrid overall={report.candidate.overall.rate():.3f} cached={report.baseline.overall.rate():.3f}; "
        f"endgame hybrid={report.candidate.endgame.rate():.3f} cached={report.baseline.endgame.rate():.3f}"
    )
    return GateResult("C PMR regression", end_ok and global_ok, detail)


def write_report_md(report: Report, output_path: Path) -> None:
    lines = ["# Hybrid submission validation", ""]
    for gate in report.gates:
        status = "PASS" if gate.passed else "FAIL"
        lines.append(f"- [{status}] {gate.name}: {gate.detail}")
    lines.append("")
    lines.append(f"**RESULT: {'PASS' if report.passed else 'FAIL'}**")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    hybrid_path: Path,
    cached_path: Path,
    *,
    skip_pmr: bool,
    skip_slow: bool,
    bitboard_samples: int,
) -> Report:
    report = Report()
    hybrid_mod = load_agent(hybrid_path)
    hybrid_mod._SOLVER_MAX_EMPTY = 18
    hybrid_agent = hybrid_mod.agent
    cached_mod = load_agent(cached_path)
    cached_agent = cached_mod.agent
    full_cfg = {"rows": 6, "columns": 7, "inarow": 4, "actTimeout": 2.0, "timeout": 2.0}

    for gate in (
        gate_a_bitboard_parity(hybrid_mod, bitboard_samples, 0),
        gate_a_opening(hybrid_agent),
        gate_a_safety(hybrid_agent, cached_agent, full_cfg, 40),
        gate_a_endgames(hybrid_agent, full_cfg),
        gate_a_losing_fallback(hybrid_mod, cached_agent, full_cfg),
    ):
        report.add(gate.name, gate.passed, gate.detail)

    if not skip_slow:
        for gate in (
            gate_b_fast_arena(hybrid_agent, cached_agent),
            gate_b_full_time(hybrid_agent, cached_agent, 20),
        ):
            report.add(gate.name, gate.passed, gate.detail)

    if not skip_pmr:
        gate = gate_c_pmr(hybrid_path, cached_path)
        report.add(gate.name, gate.passed, gate.detail)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", type=Path, default=DEFAULT_HYBRID)
    parser.add_argument("--cached", type=Path, default=DEFAULT_CACHED)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "results/validate_hybrid.md")
    parser.add_argument("--skip-pmr", action="store_true")
    parser.add_argument("--skip-slow", action="store_true")
    parser.add_argument("--bitboard-samples", type=int, default=100_000)
    args = parser.parse_args(argv)

    report = run_validation(
        args.hybrid,
        args.cached,
        skip_pmr=args.skip_pmr,
        skip_slow=args.skip_slow,
        bitboard_samples=args.bitboard_samples,
    )
    write_report_md(report, args.report)
    print("=== Hybrid validation ===")
    for gate in report.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"[{status}] {gate.name}: {gate.detail}")
    print("RESULT:", "PASS" if report.passed else "FAIL")
    print(f"Wrote {args.report}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
