"""Inject bitboard solver into a rollback submission template."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SOLVER_CODE = '''
# --- bitboard exact solver (Pascal Pons layout) ---
_BB_W = 7
_BB_H = 6
_BB_BOTTOM = tuple(1 << (7 * col) for col in range(_BB_W))
_BB_TOP = tuple(_BB_BOTTOM[col] << (_BB_H - 1) for col in range(_BB_W))


def _bb_has_won(position):
    joined = position & (position >> 7)
    if joined & (joined >> 14):
        return True
    joined = position & (position >> 1)
    if joined & (joined >> 2):
        return True
    joined = position & (position >> 6)
    if joined & (joined >> 12):
        return True
    joined = position & (position >> 8)
    if joined & (joined >> 16):
        return True
    return False


def _bb_board_to_state(board, mark):
    position = 0
    mask = 0
    for row in range(_BB_H):
        for col in range(_BB_W):
            cell = int(board[row * _BB_W + col])
            if cell == 0:
                continue
            bit = 1 << ((5 - row) + 7 * col)
            mask |= bit
            if cell == mark:
                position |= bit
    return position, mask, mark


def _bb_legal(mask):
    moves = []
    for col in range(_BB_W):
        if not (mask & _BB_TOP[col]):
            moves.append(col)
    return moves


def _bb_play(position, mask, col):
    move = mask + _BB_BOTTOM[col]
    return position | move, mask | move


def _bb_negamax(position, mask, current, alpha, beta, tt, counter, deadline):
    counter[0] += 1
    if counter[0] % 4096 == 0 and time.time() >= deadline:
        return None
    key = position | mask | (current << 42)
    cached = tt.get(key)
    if cached is not None:
        score, flag, _move = cached
        if flag == 0 or (flag < 0 and score <= alpha) or (flag > 0 and score >= beta):
            return score

    legal = _bb_legal(mask)
    if not legal:
        tt[key] = (0, 0, None)
        return 0

    best_score = -2
    flag = -1
    for col in sorted(legal, key=lambda c: (abs(c - 3), c)):
        next_position, next_mask = _bb_play(position, mask, col)
        if _bb_has_won(next_position):
            score = 1
        elif bin(next_mask).count("1") >= 42:
            score = 0
        else:
            child = _bb_negamax(
                next_mask - next_position,
                next_mask,
                3 - current,
                -beta,
                -alpha,
                tt,
                counter,
                deadline,
            )
            if child is None:
                return None
            score = -child
        if score > best_score:
            best_score = score
        alpha = max(alpha, score)
        if alpha >= beta:
            flag = 1
            break
    tt[key] = (best_score, flag, None)
    return best_score


def _bb_solve_root(board, mark, deadline):
    if time.time() >= deadline:
        return None
    position, mask, current = _bb_board_to_state(board, mark)
    tt = {}
    counter = [0]
    scores = {}
    for col in _bb_legal(mask):
        if time.time() >= deadline:
            return None
        next_position, next_mask = _bb_play(position, mask, col)
        if _bb_has_won(next_position):
            scores[col] = 1
            continue
        if bin(next_mask).count("1") >= 42:
            scores[col] = 0
            continue
        child = _bb_negamax(
            next_mask - next_position,
            next_mask,
            3 - current,
            -1,
            1,
            tt,
            counter,
            deadline,
        )
        if child is None:
            return None
        scores[col] = -child
    if not scores:
        return 0, ()
    best = max(scores.values())
    if best < 0:
        return best, ()
    best_moves = tuple(sorted(col for col, value in scores.items() if value == best))
    return best, best_moves


def _exact_action(board, mark, deadline):
    solved = _bb_solve_root(board, mark, deadline)
    if solved is None:
        return None
    score, moves = solved
    if score >= 0 and moves:
        return int(moves[0])
    return None
'''

NEW_AGENT = '''
def agent(observation, configuration):
    cfg = _cfg(configuration)
    rows, columns, inarow = cfg["rows"], cfg["columns"], cfg["inarow"]
    board, mark = _obs_board_mark(observation)
    action = _tactical(board, mark, rows, columns, inarow)
    if action is not None:
        return int(action)
    total_budget = max(0.05, min(1.95, cfg["timeout"] * 0.88))
    deadline = time.time() + total_budget
    solver_deadline = time.time() + total_budget * 0.5
    if rows == 6 and columns == 7 and inarow == 4:
        exact = _exact_action(board, mark, solver_deadline)
        if exact is not None and _mask(board, rows, columns)[exact]:
            return int(exact)
    action = _search(board, mark, rows, columns, inarow, deadline)
    if not _mask(board, rows, columns)[action]:
        legal = np.flatnonzero(_mask(board, rows, columns))
        return int(legal[0]) if legal.size else 0
    return int(action)
'''


def build_solver_submission(template_path: Path, output_path: Path) -> Path:
    text = template_path.read_text(encoding="utf-8")
    text = re.sub(
        r"# note:.*\n",
        "# note: gen8 rollback + bitboard exact solver (50% budget)\n",
        text,
        count=1,
    )
    text = text.replace("# tag: rollback_gen8", "# tag: solver_gen8")

    pattern = re.compile(r"def agent\(observation, configuration\):[\s\S]*\Z")
    if not pattern.search(text):
        raise ValueError("template missing agent()")
    text = pattern.sub(SOLVER_CODE + "\n" + NEW_AGENT.strip() + "\n", text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("submission/submission_alphazero_rollback_gen8.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission/submission_alphazero_solver_gen8.py"),
    )
    args = parser.parse_args()
    path = build_solver_submission(args.template, args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
