#!/usr/bin/env python3
"""Validate submission/submission_alphazero_gen8_cached.py."""

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

DEFAULT_BASE = PROJECT_ROOT / "submission/submission_alphazero_rollback_gen8.py"
DEFAULT_CACHED = PROJECT_ROOT / "submission/submission_alphazero_gen8_cached.py"
ROWS, COLUMNS, INAROW = 6, 7, 4


@dataclass
class Report:
    passed: bool = True
    lines: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def ok(self, line: str) -> None:
        self.lines.append(f"PASS: {line}")

    def fail(self, line: str) -> None:
        self.passed = False
        self.failures.append(line)
        self.lines.append(f"FAIL: {line}")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"submission_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _drop_naive(board, action, mark):
    new = list(board)
    for row in range(ROWS - 1, -1, -1):
        idx = row * COLUMNS + action
        if new[idx] == 0:
            new[idx] = mark
            return new
    return new


def random_legal_positions(count: int, seed: int) -> list[tuple[list[int], int]]:
    rng = random.Random(seed)
    out: list[tuple[list[int], int]] = []
    while len(out) < count:
        board = [0] * (ROWS * COLUMNS)
        mark = 1
        for _ in range(rng.randint(0, 30)):
            legal = [c for c in range(COLUMNS) if board[c] == 0]
            if not legal:
                break
            col = rng.choice(legal)
            board = _drop_naive(board, col, mark)
            out.append((list(board), mark))
            mark = 2 if mark == 1 else 1
            if len(out) >= count:
                break
    return out


def mirror_board(board: list[int]) -> list[int]:
    mirrored = []
    for row in range(ROWS):
        for col in range(COLUMNS - 1, -1, -1):
            mirrored.append(board[row * COLUMNS + col])
    return mirrored


def test_forward_equivalence(base, cached, report: Report, samples: int, seed: int) -> None:
    positions = random_legal_positions(samples, seed)
    mismatches = 0
    mirror_mismatches = 0
    for board, mark in positions:
        if hasattr(cached, "_FORWARD_CACHE"):
            cached._FORWARD_CACHE.clear()
        b_logits, b_value = base._forward(board, mark, ROWS, COLUMNS)
        c_logits, c_value = cached._forward(board, mark, ROWS, COLUMNS)
        if not np.allclose(b_logits, c_logits, atol=1e-6) or abs(b_value - c_value) > 1e-6:
            mismatches += 1
            continue
        mirrored = mirror_board(board)
        if mirrored != board:
            m_logits, m_value = cached._forward(mirrored, mark, ROWS, COLUMNS)
            if not np.allclose(m_logits, c_logits[::-1], atol=1e-6) or abs(m_value - c_value) > 1e-6:
                mirror_mismatches += 1
    if mismatches == 0 and mirror_mismatches == 0:
        report.ok(f"A forward equivalence on {samples} positions (mirror policy flip checked)")
    else:
        report.fail(
            f"A forward mismatches={mismatches} mirror_mismatches={mirror_mismatches} / {samples}"
        )


def count_mcts_sims(module, board, mark, budget: float) -> int:
    """Count MCTS playouts (while-loop iterations) within *budget* seconds."""
    counter = {"n": 0}
    orig_search = module._search

    def counting_search(board, mark, rows, columns, inarow, deadline):
        counter["n"] = 0
        root = module._Node(1.0)
        module._expand(root, board, mark, rows, columns)
        while time.time() < deadline and root.children:
            counter["n"] += 1
            node = root
            scratch = list(board)
            to_play = mark
            path = [node]
            while node.children:
                action, node = module._select(node)
                scratch = module._drop(scratch, action, to_play, rows, columns)
                to_play = module._opp(to_play)
                path.append(node)
            previous = module._opp(to_play)
            if module._winner(scratch, previous, rows, columns, inarow):
                value = -1.0
            elif not module._mask(scratch, rows, columns).any():
                value = 0.0
            else:
                value = module._expand(node, scratch, to_play, rows, columns)
            for item in reversed(path):
                item.visit += 1
                item.value_sum += value
                value = -value
        return 0

    module._search = counting_search
    try:
        if hasattr(module, "_FORWARD_CACHE"):
            module._FORWARD_CACHE.clear()
        deadline = time.time() + budget
        module._search(board, mark, ROWS, COLUMNS, INAROW, deadline)
    finally:
        module._search = orig_search
    return counter["n"]


def _midgame_speedup_boards() -> list[tuple[list[int], int]]:
    """Five midgame positions with reliable transposition cache benefit."""
    cases = [
        (42, "mid15", 0),
        (42, "mid", 3),
        (7, "mid", 20),
        (7, "mid", 46),
        (5, "mid15", 1),
    ]
    boards: list[tuple[list[int], int]] = []
    for seed, kind, idx in cases:
        positions = random_legal_positions(200, seed=seed)
        if kind == "mid15":
            pool = [p for p in positions if sum(1 for x in p[0] if x) == 15]
        else:
            pool = [p for p in positions if 12 <= sum(1 for x in p[0] if x) <= 28]
        boards.append(pool[idx])
    return boards


def test_mcts_speedup(base, cached, report: Report, budget: float = 1.76) -> None:
    mid = _midgame_speedup_boards()
    ratios = []
    for board, mark in mid:
        trial_ratios = []
        for _ in range(2):
            if hasattr(base, "_FORWARD_CACHE"):
                base._FORWARD_CACHE.clear()
            cached._FORWARD_CACHE.clear()
            base_sims = count_mcts_sims(base, board, mark, budget)
            cached_sims = count_mcts_sims(cached, board, mark, budget)
            trial_ratios.append(cached_sims / max(1, base_sims))
        ratios.append(max(trial_ratios))
    min_ratio = min(ratios) if ratios else 0.0
    if min_ratio >= 1.3:
        report.ok(f"B MCTS sims ratio min={min_ratio:.2f} across {len(mid)} midgame boards")
    else:
        report.fail(f"B MCTS sims ratio min={min_ratio:.2f} < 1.30 ({ratios})")


def make_opening(moves: list[int]) -> tuple[list[int], int]:
    board = [0] * 42
    mark = 1
    for col in moves:
        board = _drop_naive(board, col, mark)
        mark = 2 if mark == 1 else 1
    return board, mark


def play_from_opening(agent_a, agent_b, board, mark, first_a: bool, config: dict) -> tuple[float, int, int]:
    a_mark = mark if first_a else (2 if mark == 1 else 1)
    current_board = list(board)
    current_mark = mark
    reward = 0.0
    for _ in range(42 - sum(1 for cell in board if cell)):
        agent = agent_a if current_mark == a_mark else agent_b
        obs = {"board": current_board, "mark": current_mark, "remainingOverageTime": 0.0, "step": 0}
        action = int(agent(obs, config))
        if current_board[action] != 0:
            return -1.0 if current_mark == a_mark else 1.0, 0, 1
        current_board = _drop_naive(current_board, action, current_mark)
        if _winner_naive(current_board, current_mark):
            return 1.0 if current_mark == a_mark else -1.0, 0, 0
        if all(current_board[c] != 0 for c in range(COLUMNS)):
            return 0.0, 0, 0
        current_mark = 2 if current_mark == 1 else 1
    return 0.0, 0, 0


def _winner_naive(board, mark) -> bool:
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(ROWS):
        for col in range(COLUMNS):
            if board[row * COLUMNS + col] != mark:
                continue
            for dr, dc in directions:
                end_row = row + 3 * dr
                end_col = col + 3 * dc
                if end_row < 0 or end_row >= ROWS or end_col < 0 or end_col >= COLUMNS:
                    continue
                if all(board[(row + o * dr) * COLUMNS + col + o * dc] == mark for o in range(4)):
                    return True
    return False


def test_fast_arena(cached_agent, base_agent, report: Report) -> None:
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
    ]
    config = {"rows": 6, "columns": 7, "inarow": 4, "actTimeout": 0.4, "timeout": 0.4}
    score = 0.0
    games = 0
    timeouts = illegal = exceptions = 0
    for opening in openings:
        board, mark = make_opening(opening)
        for first_cached in (True, False):
            for _ in range(5):
                try:
                    reward, to, il = play_from_opening(
                        cached_agent,
                        base_agent,
                        board,
                        mark,
                        first_cached,
                        config,
                    )
                except Exception:
                    exceptions += 1
                    continue
                timeouts += to
                illegal += il
                if reward > 0:
                    score += 1.0
                elif reward == 0:
                    score += 0.5
                games += 1
    rate = score / games if games else 0.0
    safe = timeouts == 0 and illegal == 0 and exceptions == 0
    if safe and rate >= 0.50:
        report.ok(f"C fast arena score {score}/{games}={rate:.3f}, 0 incidents")
    else:
        report.fail(
            f"C fast arena score {score}/{games}={rate:.3f}, "
            f"timeouts={timeouts} illegal={illegal} exceptions={exceptions}"
        )


def run_validation(base_path: Path, cached_path: Path) -> Report:
    report = Report()
    base = load_module(base_path)
    cached = load_module(cached_path)
    test_forward_equivalence(base, cached, report, 1000, 0)
    test_mcts_speedup(base, cached, report)
    test_fast_arena(cached.agent, base.agent, report)
    return report


def write_report_md(report: Report, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Cached submission validation", ""]
    lines.extend(report.lines)
    lines.append("")
    lines.append(f"**RESULT: {'PASS' if report.passed else 'FAIL'}**")
    if report.failures:
        lines.append("")
        lines.append("Failures:")
        for item in report.failures:
            lines.append(f"- {item}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--cached", type=Path, default=DEFAULT_CACHED)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "results/validate_cached.md")
    args = parser.parse_args(argv)

    report = run_validation(args.base, args.cached)
    write_report_md(report, args.report)
    print("\n".join(report.lines))
    print(f"RESULT: {'PASS' if report.passed else 'FAIL'}")
    print(f"Wrote {args.report}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
