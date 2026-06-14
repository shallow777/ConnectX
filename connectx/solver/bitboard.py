"""Bitboard Connect4 exact solver (Pascal Pons style layout)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

WIDTH = 7
HEIGHT = 6
BOARD_CELLS = WIDTH * HEIGHT

# Each column uses 7 bits: 6 playable cells + 1 separator above.
_COLUMN_MASK = tuple(((1 << HEIGHT) - 1) << (HEIGHT + 1) * col for col in range(WIDTH))
_BOTTOM_MASK = tuple(1 << ((HEIGHT + 1) * col) for col in range(WIDTH))
_TOP_MASK = tuple(_BOTTOM_MASK[col] << (HEIGHT - 1) for col in range(WIDTH))


def _has_won(position: int) -> bool:
    # Horizontal.
    joined = position & (position >> (HEIGHT + 1))
    if joined & (joined >> (2 * (HEIGHT + 1))):
        return True
    # Vertical.
    joined = position & (position >> 1)
    if joined & (joined >> 2):
        return True
    # Diagonal \ .
    joined = position & (position >> HEIGHT)
    if joined & (joined >> (2 * HEIGHT)):
        return True
    # Diagonal / .
    joined = position & (position >> (HEIGHT + 2))
    if joined & (joined >> (2 * (HEIGHT + 2))):
        return True
    return False


def board_to_state(
    board: list[int],
    mark: int,
    *,
    rows: int = HEIGHT,
    columns: int = WIDTH,
) -> tuple[int, int, int]:
    """Return (position, mask, current_mark) for the player to move."""
    if rows != HEIGHT or columns != WIDTH:
        raise ValueError("bitboard solver only supports standard 6x7 connect-4")

    position = 0
    mask = 0
    for row in range(rows):
        for col in range(columns):
            cell = int(board[row * columns + col])
            bit = 1 << ((HEIGHT - 1 - row) + (HEIGHT + 1) * col)
            if cell == 0:
                continue
            mask |= bit
            if cell == mark:
                position |= bit
    return position, mask, mark


def state_to_board(
    position: int,
    mask: int,
    current_mark: int,
    *,
    rows: int = HEIGHT,
    columns: int = WIDTH,
) -> tuple[list[int], int]:
    board = [0] * (rows * columns)
    other = 2 if current_mark == 1 else 1
    for row in range(rows):
        for col in range(columns):
            bit = 1 << ((HEIGHT - 1 - row) + (HEIGHT + 1) * col)
            if mask & bit:
                board[row * columns + col] = current_mark if position & bit else other
    return board, current_mark


def count_pieces(board: list[int]) -> int:
    return sum(1 for cell in board if cell)


def count_empty(board: list[int]) -> int:
    return sum(1 for cell in board if cell == 0)


def classify_phase(board: list[int], *, rows: int = HEIGHT, columns: int = WIDTH) -> str:
    pieces = count_pieces(board)
    empty = rows * columns - pieces
    if pieces <= 10:
        return "opening"
    if empty <= 16:
        return "endgame"
    return "midgame"


@dataclass(frozen=True)
class SolveResult:
    score: int
    best_moves: tuple[int, ...]
    nodes: int
    completed: bool


class BitboardSolver:
    """Negamax + alpha-beta + transposition table on bitboards."""

    __slots__ = ("_tt", "_node_counter", "_deadline", "_check_every")

    def __init__(self, *, table_size: int = 1 << 20, check_every: int = 4096) -> None:
        self._tt: dict[int, tuple[int, int, int | None]] = {}
        self._node_counter = 0
        self._deadline: float | None = None
        self._check_every = int(check_every)

    def clear_table(self) -> None:
        self._tt.clear()

    def _timed_out(self) -> bool:
        if self._deadline is None:
            return False
        self._node_counter += 1
        if self._node_counter % self._check_every != 0:
            return False
        import time

        return time.time() >= self._deadline

    def legal_moves(self, mask: int) -> list[int]:
        moves = []
        for col in range(WIDTH):
            if not (mask & _TOP_MASK[col]):
                moves.append(col)
        return moves

    def play_move(self, position: int, mask: int, col: int) -> tuple[int, int]:
        move = mask + _BOTTOM_MASK[col]
        return position | move, mask | move

    def solve_unlimited(self, board: list[int], mark: int) -> SolveResult:
        position, mask, current = board_to_state(board, mark)
        self._deadline = None
        self._node_counter = 0
        score, moves, nodes, completed = self._solve_root(position, mask, current)
        return SolveResult(score=score, best_moves=moves, nodes=nodes, completed=completed)

    def solve_timed(
        self,
        board: list[int],
        mark: int,
        deadline: float,
    ) -> SolveResult | None:
        position, mask, current = board_to_state(board, mark)
        self._deadline = deadline
        self._node_counter = 0
        score, moves, nodes, completed = self._solve_root(position, mask, current)
        if not completed:
            return None
        return SolveResult(score=score, best_moves=moves, nodes=nodes, completed=True)

    def move_scores(self, board: list[int], mark: int) -> dict[int, int]:
        position, mask, current = board_to_state(board, mark)
        self._deadline = None
        self._node_counter = 0
        scores: dict[int, int] = {}
        for col in self.legal_moves(mask):
            next_position, next_mask = self.play_move(position, mask, col)
            if _has_won(next_position):
                scores[col] = 1
                continue
            if bin(next_mask).count("1") >= BOARD_CELLS:
                scores[col] = 0
                continue
            # Opponent to move: negate child score.
            child_score, _, completed = self._negamax(next_mask - next_position, next_mask, 3 - current, -1, 1)
            if not completed:
                continue
            scores[col] = -child_score
        return scores

    def optimal_moves(self, board: list[int], mark: int) -> tuple[int, ...]:
        scores = self.move_scores(board, mark)
        if not scores:
            return tuple()
        best = max(scores.values())
        return tuple(sorted(col for col, value in scores.items() if value == best))

    def _solve_root(
        self,
        position: int,
        mask: int,
        current: int,
    ) -> tuple[int, tuple[int, ...], int, bool]:
        scores: dict[int, int] = {}
        for col in self.legal_moves(mask):
            if self._timed_out():
                return 0, tuple(), self._node_counter, False
            next_position, next_mask = self.play_move(position, mask, col)
            if _has_won(next_position):
                scores[col] = 1
                continue
            if bin(next_mask).count("1") >= BOARD_CELLS:
                scores[col] = 0
                continue
            child_score, _, completed = self._negamax(
                next_mask - next_position,
                next_mask,
                3 - current,
                -1,
                1,
            )
            if not completed:
                return 0, tuple(), self._node_counter, False
            scores[col] = -child_score

        if not scores:
            return 0, tuple(), self._node_counter, True
        best = max(scores.values())
        best_moves = tuple(sorted(col for col, value in scores.items() if value == best))
        return best, best_moves, self._node_counter, True

    def _negamax(
        self,
        position: int,
        mask: int,
        current: int,
        alpha: int,
        beta: int,
    ) -> tuple[int, int | None, bool]:
        if self._timed_out():
            return 0, None, False

        key = position | mask | (current << 42)
        tt_entry = self._tt.get(key)
        if tt_entry is not None:
            tt_score, flag, tt_move = tt_entry
            if flag == 0 or (flag < 0 and tt_score <= alpha) or (flag > 0 and tt_score >= beta):
                return tt_score, tt_move, True

        legal = self.legal_moves(mask)
        if not legal:
            self._tt[key] = (0, 0, None)
            return 0, None, True

        best_move: int | None = None
        best_score = -2
        flag = -1

        for col in self._move_order(mask, legal):
            if self._timed_out():
                return 0, None, False
            next_position, next_mask = self.play_move(position, mask, col)
            if _has_won(next_position):
                score = 1
            elif bin(next_mask).count("1") >= BOARD_CELLS:
                score = 0
            else:
                child_score, _, completed = self._negamax(
                    next_mask - next_position,
                    next_mask,
                    3 - current,
                    -beta,
                    -alpha,
                )
                if not completed:
                    return 0, None, False
                score = -child_score

            if score > best_score:
                best_score = score
                best_move = col
            alpha = max(alpha, score)
            if alpha >= beta:
                flag = 1
                break

        self._tt[key] = (best_score, flag, best_move)
        return best_score, best_move, True

    def _move_order(self, mask: int, legal: list[int]) -> list[int]:
        center_first = sorted(legal, key=lambda col: (abs(col - WIDTH // 2), col))
        return center_first


def sample_selfplay_positions(
    agent_a: Callable[[dict, dict], int],
    agent_b: Callable[[dict, dict], int],
    *,
    games: int = 80,
    min_midgame: int = 500,
    rows: int = HEIGHT,
    columns: int = WIDTH,
    inarow: int = 4,
    timeout: float = 2.0,
) -> list[tuple[list[int], int, str]]:
    """Play games and collect opening/midgame/endgame positions."""
    positions: list[tuple[list[int], int, str]] = []
    midgame_count = 0
    config = {
        "rows": rows,
        "columns": columns,
        "inarow": inarow,
        "actTimeout": timeout,
        "timeout": timeout,
    }

    for game_idx in range(games):
        board = [0] * (rows * columns)
        mark = 1
        agents = (agent_a, agent_b) if game_idx % 2 == 0 else (agent_b, agent_a)
        for _step in range(rows * columns):
            phase = classify_phase(board, rows=rows, columns=columns)
            if phase == "midgame":
                midgame_count += 1
            positions.append((list(board), mark, phase))

            obs = {"board": board, "mark": mark, "remainingOverageTime": 0.0, "step": _step}
            action = int(agents[mark - 1](obs, config))
            if board[action] != 0:
                # Illegal move ends data collection for this game.
                break
            placed = False
            for row in range(rows - 1, -1, -1):
                idx = row * columns + action
                if board[idx] == 0:
                    board[idx] = mark
                    placed = True
                    break
            if not placed:
                break

            if _is_terminal(board, mark, rows, columns, inarow):
                break
            mark = 2 if mark == 1 else 1

        if midgame_count >= min_midgame:
            break

    return positions


def _is_terminal(board: list[int], last_mark: int, rows: int, columns: int, inarow: int) -> bool:
    if _winner(board, last_mark, rows, columns, inarow):
        return True
    return all(board[col] != 0 for col in range(columns))


def _winner(board: list[int], mark: int, rows: int, columns: int, inarow: int) -> bool:
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
