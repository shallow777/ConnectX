"""Build submission_alphazero_hybrid_gen8.py from cached base + v2 solver + opening book."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

from connectx.submission.make_submission import _extract_template_body

HYBRID_AGENT_BLOCK = r'''
# --- bitboard exact solver (Pascal Pons layout, hybrid integration) ---
_BB_W = 7
_BB_H = 6
_BB_BOTTOM = tuple(1 << (7 * col) for col in range(_BB_W))
_BB_TOP = tuple(_BB_BOTTOM[col] << (_BB_H - 1) for col in range(_BB_W))
_SOLVER_MAX_EMPTY = 18


def _solver_backoff_on_timeout(empty):
    global _SOLVER_MAX_EMPTY
    limit = max(0, int(empty) - 2)
    if limit < _SOLVER_MAX_EMPTY:
        _SOLVER_MAX_EMPTY = limit


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


def _bb_state_to_board(position, mask, mark):
    other = _opp(mark)
    board = [0] * (_BB_W * _BB_H)
    for row in range(_BB_H):
        for col in range(_BB_W):
            bit = 1 << ((5 - row) + 7 * col)
            if mask & bit:
                board[row * _BB_W + col] = mark if position & bit else other
    return board


def _bb_legal(mask):
    moves = []
    for col in range(_BB_W):
        if not (mask & _BB_TOP[col]):
            moves.append(col)
    return moves


def _bb_column_mask(col):
    return ((1 << _BB_H) - 1) << (7 * col)


def _bb_stone_bit(mask, col):
    occupied = mask & _bb_column_mask(col)
    if occupied == 0:
        return _BB_BOTTOM[col]
    high = occupied.bit_length() - 1
    return 1 << (high + 1)


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


def _bb_classify_root(board, mark, deadline):
    if time.time() >= deadline:
        return "unknown", None
    position, mask, current = _bb_board_to_state(board, mark)
    legal = _bb_legal(mask)
    if not legal:
        return "filter", None
    wins = []
    draws = []
    tt = {}
    counter = [0]
    for col in sorted(legal, key=lambda c: (abs(c - 3), c)):
        if time.time() >= deadline:
            if wins:
                return "forced", int(wins[0])
            return "unknown", None
        next_position, next_mask = _bb_play(position, mask, col)
        if _bb_has_won(next_position):
            wins.append(col)
            continue
        if bin(next_mask).count("1") >= 42:
            draws.append(col)
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
            if wins:
                return "forced", int(wins[0])
            return "unknown", None
        score = -child
        if score == 1:
            wins.append(col)
        elif score == 0:
            draws.append(col)
    if wins:
        return "filter", tuple(sorted(wins))
    if draws:
        return "filter", tuple(sorted(draws))
    return "filter", None


def _opening_hardcode(board, mark, rows, columns, inarow):
    if rows != 6 or columns != 7 or inarow != 4:
        return None
    pieces = sum(1 for cell in board if cell)
    if pieces > 1:
        return None
    if pieces == 0:
        return 3 if _mask(board, rows, columns)[3] else None
    if mark != 2:
        return None
    first_col = next(
        col for col in range(columns) if any(board[row * columns + col] for row in range(rows))
    )
    for adj in sorted((first_col - 1, first_col + 1), key=lambda c: abs(c - 3)):
        if 0 <= adj < columns and _mask(board, rows, columns)[adj]:
            return int(adj)
    return None


def _drop(board, action, mark, rows, columns):
    if rows == _BB_H and columns == _BB_W:
        position, mask, _ = _bb_board_to_state(board, mark)
        stone = _bb_stone_bit(mask, int(action))
        return _bb_state_to_board(position | stone, mask | stone, mark)
    new = list(board)
    for row in range(rows - 1, -1, -1):
        idx = row * columns + action
        if new[idx] == 0:
            new[idx] = mark
            return new
    return new


def _winner(board, mark, rows, columns, inarow):
    if rows == _BB_H and columns == _BB_W and inarow == 4:
        position, _, _ = _bb_board_to_state(board, mark)
        return _bb_has_won(position)
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
                ok = True
                for offset in range(inarow):
                    if board[(row + offset * dr) * columns + col + offset * dc] != mark:
                        ok = False
                        break
                if ok:
                    return True
    return False


def _search(board, mark, rows, columns, inarow, deadline, root_allowed=None):
    root = _Node(1.0)
    _expand(root, board, mark, rows, columns)
    if root_allowed:
        allowed = set(root_allowed)
        for action in list(root.children.keys()):
            if action not in allowed:
                del root.children[action]
    while time.time() < deadline and root.children:
        node = root
        scratch = list(board)
        to_play = mark
        path = [node]
        while node.children:
            action, node = _select(node)
            scratch = _drop(scratch, action, to_play, rows, columns)
            to_play = _opp(to_play)
            path.append(node)
        previous = _opp(to_play)
        if _winner(scratch, previous, rows, columns, inarow):
            value = -1.0
        elif not _mask(scratch, rows, columns).any():
            value = 0.0
        else:
            value = _expand(node, scratch, to_play, rows, columns)
        for item in reversed(path):
            item.visit += 1
            item.value_sum += value
            value = -value
    if not root.children:
        legal = np.flatnonzero(_mask(board, rows, columns))
        if root_allowed:
            for action in root_allowed:
                if action < len(legal) and legal[action]:
                    return int(action)
        return int(legal[0]) if legal.size else 0
    return max(root.children.items(), key=lambda kv: kv[1].visit)[0]


def agent(observation, configuration):
    global _SOLVER_MAX_EMPTY
    cfg = _cfg(configuration)
    rows, columns, inarow = cfg["rows"], cfg["columns"], cfg["inarow"]
    board, mark = _obs_board_mark(observation)
    if sum(1 for cell in board if cell) == 0:
        _SOLVER_MAX_EMPTY = 18
    action = _opening_hardcode(board, mark, rows, columns, inarow)
    if action is not None:
        return int(action)
    action = _tactical(board, mark, rows, columns, inarow)
    if action is not None:
        return int(action)
    total_budget = max(0.05, min(1.95, cfg["timeout"] * 0.88))
    start = time.time()
    deadline = start + total_budget
    root_allowed = None
    empty = sum(1 for cell in board if cell == 0)
    if rows == 6 and columns == 7 and inarow == 4 and empty <= _SOLVER_MAX_EMPTY and total_budget >= 0.15:
        solver_deadline = start + total_budget * 0.30
        mode, payload = _bb_classify_root(board, mark, solver_deadline)
        if mode == "forced":
            if _mask(board, rows, columns)[payload]:
                return int(payload)
        elif mode == "unknown":
            _solver_backoff_on_timeout(empty)
        elif mode == "filter" and payload:
            root_allowed = payload
    action = _search(board, mark, rows, columns, inarow, deadline, root_allowed)
    if not _mask(board, rows, columns)[action]:
        legal = np.flatnonzero(_mask(board, rows, columns))
        return int(legal[0]) if legal.size else 0
    return int(action)
'''


def build_hybrid_submission(template_path: Path, output_path: Path) -> Path:
    body = _extract_template_body(template_path)
    body_text = "\n".join(body)

    body_text = re.sub(
        r"def _drop\(board, action, mark, rows, columns\):[\s\S]*?return new\n\n\n",
        "",
        body_text,
        count=1,
    )
    body_text = re.sub(
        r"def _winner\(board, mark, rows, columns, inarow\):[\s\S]*?return False\n\n\n",
        "",
        body_text,
        count=1,
    )
    body_text = re.sub(
        r"def _search\(board, mark, rows, columns, inarow, deadline\):[\s\S]*?return max\(root\.children\.items\(\), key=lambda kv: kv\[1\]\.visit\)\[0\]\n\n\n",
        "",
        body_text,
        count=1,
    )
    body_text = re.sub(
        r"def agent\(observation, configuration\):[\s\S]*\Z",
        "",
        body_text,
        count=1,
    )

    header = [
        "# Kaggle ConnectX submission",
        "# algorithm: alphazero",
        "# board: 6x7 connect-4",
        "# checkpoint: runs/alphazero_push/checkpoints/generation_0008_accepted.pt",
        "# tag: hybrid_gen8",
        f"# exported_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "# note: gen8 cached + TB-style solver + opening hardcode",
        "",
    ]

    template_text = template_path.read_text(encoding="utf-8")
    weights_line = next(line for line in template_text.splitlines() if line.startswith("WEIGHTS_B64"))

    content = "\n".join(header + [weights_line, ""] + body_text.splitlines() + [HYBRID_AGENT_BLOCK.strip(), ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("submission/submission_alphazero_gen8_cached.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission/submission_alphazero_hybrid_gen8.py"),
    )
    args = parser.parse_args()
    path = build_hybrid_submission(args.template, args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
