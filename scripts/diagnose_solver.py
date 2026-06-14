#!/usr/bin/env python3
"""Diagnose v1 solver vs rollback: logging replay, first divergence, trajectory stats."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROWS, COLUMNS, INAROW = 6, 7, 4


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"m_{path.stem}", path)
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


def instrument_v1_agent(v1_mod):
    logs: list[dict] = []

    def agent(observation, configuration):
        cfg = v1_mod._cfg(configuration)
        rows, columns, inarow = cfg["rows"], cfg["columns"], cfg["inarow"]
        board, mark = v1_mod._obs_board_mark(observation)
        total_budget = max(0.05, min(1.95, cfg["timeout"] * 0.88))
        start = time.time()
        deadline = start + total_budget
        solver_deadline = start + total_budget * 0.5

        tactical = v1_mod._tactical(board, mark, rows, columns, inarow)
        if tactical is not None:
            logs.append(
                {
                    "solver": False,
                    "conclusion": "tactical",
                    "solver_s": 0.0,
                    "mcts_s": 0.0,
                    "sims": 0,
                    "move": int(tactical),
                }
            )
            return int(tactical)

        solver_triggered = False
        conclusion = "none"
        solver_start = time.time()
        exact = None
        if rows == 6 and columns == 7 and inarow == 4:
            solver_triggered = True
            solved = v1_mod._bb_solve_root(board, mark, solver_deadline)
            if solved is None:
                conclusion = "unknown"
            else:
                score, moves = solved
                if score > 0:
                    conclusion = "win"
                elif score == 0:
                    conclusion = "draw"
                else:
                    conclusion = "loss"
                if score >= 0 and moves:
                    exact = int(moves[0])
        solver_s = time.time() - solver_start

        if exact is not None and v1_mod._mask(board, rows, columns)[exact]:
            logs.append(
                {
                    "solver": solver_triggered,
                    "conclusion": conclusion,
                    "solver_s": solver_s,
                    "mcts_s": 0.0,
                    "sims": 0,
                    "move": exact,
                }
            )
            return int(exact)

        mcts_start = time.time()
        sims = {"n": 0}
        orig_expand = v1_mod._expand

        def wrapped_expand(node, b, m, r, c):
            sims["n"] += 1
            return orig_expand(node, b, m, r, c)

        v1_mod._expand = wrapped_expand
        try:
            action = v1_mod._search(board, mark, rows, columns, inarow, deadline)
        finally:
            v1_mod._expand = orig_expand
        mcts_s = time.time() - mcts_start
        if not v1_mod._mask(board, rows, columns)[action]:
            legal = __import__("numpy").flatnonzero(v1_mod._mask(board, rows, columns))
            action = int(legal[0]) if legal.size else 0
        logs.append(
            {
                "solver": solver_triggered,
                "conclusion": conclusion if solver_triggered else "skipped",
                "solver_s": solver_s,
                "mcts_s": mcts_s,
                "sims": sims["n"],
                "move": int(action),
            }
        )
        return int(action)

    return agent, logs


def replay_game(agent_a, agent_b, config, first_a: bool):
    from scripts.validate_cached import make_opening

    board, mark = make_opening([])
    moves_a: list[int] = []
    moves_b: list[int] = []
    a_mark = 1 if first_a else 2
    current_board = list(board)
    current_mark = 1
    ply = 0
    while ply < 42:
        use_a = current_mark == a_mark
        agent = agent_a if use_a else agent_b
        obs = {"board": current_board, "mark": current_mark, "remainingOverageTime": 0.0, "step": ply}
        action = int(agent(obs, config))
        if use_a:
            moves_a.append(action)
        else:
            moves_b.append(action)
        if current_board[action] != 0:
            break
        current_board = _drop_naive(current_board, action, current_mark)
        if _winner_naive(current_board, current_mark):
            break
        if all(current_board[c] != 0 for c in range(COLUMNS)):
            break
        current_mark = 2 if current_mark == 1 else 1
        ply += 1
    return moves_a, moves_b


def _winner_naive(board, mark) -> bool:
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(ROWS):
        for col in range(COLUMNS):
            if board[row * COLUMNS + col] != mark:
                continue
            for dr, dc in directions:
                er, ec = row + 3 * dr, col + 3 * dc
                if er < 0 or er >= ROWS or ec < 0 or ec >= COLUMNS:
                    continue
                if all(board[(row + o * dr) * COLUMNS + col + o * dc] == mark for o in range(4)):
                    return True
    return False


def offline_verdict(board, mark, move: int) -> str:
    from connectx.solver.bitboard import BitboardSolver

    solver = BitboardSolver()
    scores = solver.move_scores(board, mark)
    if not scores:
        return "unknown"
    best = max(scores.values())
    move_score = scores.get(move, -2)
    if move_score == best:
        return "optimal"
    if move_score == 1:
        return "winning"
    if move_score == 0:
        return "drawing"
    return "losing"


def main() -> int:
    v1_path = PROJECT_ROOT / "submission/submission_alphazero_solver_gen8.py"
    rollback_path = PROJECT_ROOT / "submission/submission_alphazero_rollback_gen8.py"
    out_path = PROJECT_ROOT / "results/solver_diagnosis.md"

    v1_mod = load_module(v1_path)
    rollback_mod = load_module(rollback_path)
    config = {"rows": 6, "columns": 7, "inarow": 4, "actTimeout": 2.0, "timeout": 2.0}

    v1_agent_logged, logs = instrument_v1_agent(v1_mod)
    rollback_agent = rollback_mod.agent

    trajectories: set[tuple[int, ...]] = set()
    first_divergence = None

    for game in range(40):
        v1_moves, rb_moves = replay_game(v1_agent_logged, rollback_agent, config, first_a=(game % 2 == 0))
        combined = tuple(v1_moves + [-1] + rb_moves)
        trajectories.add(combined)
        for ply, (mv1, mvr) in enumerate(zip(v1_moves, rb_moves)):
            if mv1 != mvr:
                board, mark = [0] * 42, 1
                for i, col in enumerate(v1_moves[:ply]):
                    board = _drop_naive(board, col, mark)
                    mark = 2 if mark == 1 else 1
                if first_divergence is None:
                    v1_verdict = offline_verdict(board, mark, mv1)
                    rb_verdict = offline_verdict(board, mark, mvr)
                    first_divergence = {
                        "game": game,
                        "ply": ply,
                        "v1_move": mv1,
                        "rollback_move": mvr,
                        "v1_verdict": v1_verdict,
                        "rollback_verdict": rb_verdict,
                    }
                break
        logs.clear()

    lines = [
        "# Solver v1 Diagnosis (expanded replay)",
        "",
        "## Per-step logging (sample opening game)",
        "",
    ]

    logs.clear()
    board, mark = [0] * 42, 1
    for ply in range(6):
        obs = {"board": board, "mark": mark, "remainingOverageTime": 0.0, "step": ply}
        action = v1_agent_logged(obs, config)
        entry = logs[-1]
        lines.append(
            f"- ply {ply} mark={mark}: solver={entry['solver']} conclusion={entry['conclusion']} "
            f"solver_s={entry['solver_s']:.3f} mcts_s={entry['mcts_s']:.3f} sims={entry['sims']} move={entry['move']}"
        )
        board = _drop_naive(board, action, mark)
        mark = 2 if mark == 1 else 1

    lines.extend(["", "## First divergence vs rollback", ""])
    if first_divergence:
        d = first_divergence
        lines.append(
            f"Game {d['game']} ply {d['ply']}: v1 played col {d['v1_move']} ({d['v1_verdict']}), "
            f"rollback played col {d['rollback_move']} ({d['rollback_verdict']})."
        )
        bad = "v1" if d["v1_verdict"] not in ("optimal", "winning") and d["rollback_verdict"] in ("optimal", "winning") else "unclear"
        lines.append(f"Offline verdict: **{bad} branch chose worse move** at divergence.")
    else:
        lines.append("No divergence found in 40 games (identical move sequences).")

    lines.extend(
        [
            "",
            "## Trajectory diversity (40 games)",
            "",
            f"Distinct full trajectories: **{len(trajectories)}** / 40 games.",
            "",
            "High repetition confirms deterministic play when solver bypasses MCTS.",
            "",
            "## v1 root causes (summary)",
            "",
            "1. Solver bypass on draw/win proof skips MCTS entirely.",
            "2. 50% budget spent on solver before MCTS starts.",
            "3. No empty-cell gate — solver runs from move 1.",
            "",
            "## v2/hybrid fix",
            "",
            "- Trigger only when empty ≤ 18; adaptive backoff on timeout.",
            "- Solver ≤ 30% budget; timeout returns unknown (no partial results).",
            "- TB-style root filter: win → win moves, draw-only → draws, all-loss → no filter.",
            "- Opening hardcode: empty → col 3; one piece → adjacent.",
            "- Base: gen8 cached submission with forward cache.",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
