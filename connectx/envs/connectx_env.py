"""ConnectX 棋盘逻辑与 Gymnasium 环境 (board logic & Gymnasium env).

棋盘用长度为 rows*columns 的一维 list 表示, 按行优先 (row-major) 排列,
第 0 行是最上面一行; 0 = 空格, 1/2 = 两位玩家的棋子。
该表示与 Kaggle ConnectX 的 obs["board"] 完全一致, 方便直接对接线上评测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class ConnectXConfig:
    """棋盘配置: 默认是标准 6x7 四连 (connect-4); Q-learning 用 4x5 三连小棋盘."""

    rows: int = 6
    columns: int = 7
    inarow: int = 4


def opponent_mark(mark: int) -> int:
    if mark == 1:
        return 2
    if mark == 2:
        return 1
    raise ValueError(f"mark must be 1 or 2, got {mark!r}")


def _board_array(board: list[int] | tuple[int, ...] | np.ndarray, rows: int, columns: int) -> np.ndarray:
    array = np.asarray(board, dtype=np.int8)
    expected_size = rows * columns
    if array.size != expected_size:
        raise ValueError(f"board has {array.size} cells, expected {expected_size}")
    return array.reshape(rows, columns)


def encode_board(
    board: list[int] | tuple[int, ...] | np.ndarray,
    current_mark: int,
    rows: int = 6,
    columns: int = 7,
) -> np.ndarray:
    """编码为 (2, rows, columns) 张量: 通道 0 = 当前玩家棋子, 通道 1 = 对手棋子.

    始终以"当前行动方"视角编码 (current-player perspective), 这样同一个网络
    无需区分自己执先手还是后手。
    """
    array = _board_array(board, rows, columns)
    opponent = opponent_mark(current_mark)
    return np.stack(
        [
            (array == current_mark).astype(np.float32),
            (array == opponent).astype(np.float32),
        ],
        axis=0,
    )


def valid_action_mask(
    board: list[int] | tuple[int, ...] | np.ndarray,
    rows: int = 6,
    columns: int = 7,
) -> np.ndarray:
    # 顶行 (第 0 行) 还空着的列就是合法落子列 (top cell empty => column playable)
    array = _board_array(board, rows, columns)
    return (array[0] == 0).astype(np.int8)


def legal_actions(
    board: list[int] | tuple[int, ...] | np.ndarray,
    rows: int = 6,
    columns: int = 7,
) -> list[int]:
    mask = valid_action_mask(board, rows, columns)
    return [int(action) for action, valid in enumerate(mask) if valid]


def drop_piece(
    board: list[int] | np.ndarray,
    column: int,
    mark: int,
    rows: int = 6,
    columns: int = 7,
) -> int:
    """Drop one piece in-place and return the row where it landed."""
    if mark not in (1, 2):
        raise ValueError(f"mark must be 1 or 2, got {mark!r}")
    if column < 0 or column >= columns:
        raise ValueError(f"column must be in [0, {columns}), got {column!r}")

    for row in range(rows - 1, -1, -1):
        index = row * columns + column
        if int(board[index]) == 0:
            board[index] = mark
            return row
    raise ValueError(f"column {column} is full")


def next_board(
    board: list[int] | tuple[int, ...] | np.ndarray,
    column: int,
    mark: int,
    rows: int = 6,
    columns: int = 7,
) -> list[int]:
    copied = list(np.asarray(board, dtype=np.int8).reshape(rows * columns))
    drop_piece(copied, column, mark, rows, columns)
    return copied


def check_winner(
    board: list[int] | tuple[int, ...] | np.ndarray,
    mark: int,
    rows: int = 6,
    columns: int = 7,
    inarow: int = 4,
) -> bool:
    """检查 mark 是否有 inarow 连子 (horizontal / vertical / two diagonals)."""
    array = _board_array(board, rows, columns)
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))  # 右、下、右下、左下

    for row in range(rows):
        for col in range(columns):
            if int(array[row, col]) != mark:
                continue
            for dr, dc in directions:
                end_row = row + (inarow - 1) * dr
                end_col = col + (inarow - 1) * dc
                if end_row < 0 or end_row >= rows or end_col < 0 or end_col >= columns:
                    continue
                if all(int(array[row + offset * dr, col + offset * dc]) == mark for offset in range(inarow)):
                    return True
    return False


def is_draw(board: list[int] | tuple[int, ...] | np.ndarray, rows: int = 6, columns: int = 7) -> bool:
    return not bool(valid_action_mask(board, rows, columns).any())


class ConnectXEnv(gym.Env):
    """与 Kaggle ConnectX 语义兼容的双人轮流 Gymnasium 环境.

    注意这是 "轮流走子" 环境: 每次 step() 都由 current_mark 一方落子,
    奖励也发给当下落子的一方 (胜 +1 / 平 0 / 非法落子 -1 并判负)。
    观测始终从当前行动方视角编码, 自我对弈训练时两边可共用一个策略。
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        rows: int = 6,
        columns: int = 7,
        inarow: int = 4,
        illegal_reward: float = -1.0,
        win_reward: float = 1.0,
        draw_reward: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = ConnectXConfig(rows=rows, columns=columns, inarow=inarow)
        self.illegal_reward = float(illegal_reward)
        self.win_reward = float(win_reward)
        self.draw_reward = float(draw_reward)

        self.action_space = spaces.Discrete(columns)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2, rows, columns),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(low=0, high=1, shape=(columns,), dtype=np.int8),
            }
        )

        self.board: list[int] = [0] * (rows * columns)
        self.current_mark = 1
        self.move_count = 0
        self.winner = 0
        self.done = False

    @property
    def rows(self) -> int:
        return self.config.rows

    @property
    def columns(self) -> int:
        return self.config.columns

    @property
    def inarow(self) -> int:
        return self.config.inarow

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        board = options.get("board", [0] * (self.rows * self.columns))
        current_mark = int(options.get("current_mark", 1))

        _board_array(board, self.rows, self.columns)
        opponent_mark(current_mark)

        self.board = [int(cell) for cell in board]
        self.current_mark = current_mark
        self.move_count = sum(1 for cell in self.board if cell)
        self.winner = int(options.get("winner", 0))
        self.done = bool(options.get("done", False))

        return self._observation(), self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Cannot call step() on a terminated ConnectXEnv. Call reset() first.")

        action = int(action)
        actor = self.current_mark
        mask = valid_action_mask(self.board, self.rows, self.columns)
        illegal = action < 0 or action >= self.columns or int(mask[action]) == 0

        if illegal:
            self.done = True
            self.winner = opponent_mark(actor)
            info = self._info(illegal_action=True, last_action=action)
            return self._observation(), self.illegal_reward, True, False, info

        drop_piece(self.board, action, actor, self.rows, self.columns)
        self.move_count += 1

        if check_winner(self.board, actor, self.rows, self.columns, self.inarow):
            self.done = True
            self.winner = actor
            info = self._info(illegal_action=False, last_action=action)
            return self._observation(), self.win_reward, True, False, info

        if is_draw(self.board, self.rows, self.columns):
            self.done = True
            self.winner = 0
            info = self._info(illegal_action=False, last_action=action)
            return self._observation(), self.draw_reward, True, False, info

        self.current_mark = opponent_mark(actor)
        info = self._info(illegal_action=False, last_action=action)
        return self._observation(), 0.0, False, False, info

    def action_masks(self) -> np.ndarray:
        return valid_action_mask(self.board, self.rows, self.columns).astype(bool)

    def available_actions(self) -> list[int]:
        return legal_actions(self.board, self.rows, self.columns)

    def get_state(self) -> dict[str, Any]:
        return {
            "board": tuple(self.board),
            "current_mark": self.current_mark,
            "move_count": self.move_count,
            "winner": self.winner,
            "done": self.done,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        board = state["board"]
        _board_array(board, self.rows, self.columns)
        self.board = [int(cell) for cell in board]
        self.current_mark = int(state["current_mark"])
        opponent_mark(self.current_mark)
        self.move_count = int(state.get("move_count", sum(1 for cell in self.board if cell)))
        self.winner = int(state.get("winner", 0))
        self.done = bool(state.get("done", False))

    def kaggle_observation(self) -> dict[str, Any]:
        return {"board": list(self.board), "mark": self.current_mark}

    def render(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        array = _board_array(self.board, self.rows, self.columns)
        lines = [" ".join(symbols[int(cell)] for cell in row) for row in array]
        lines.append(" ".join(str(col) for col in range(self.columns)))
        return "\n".join(lines)

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "observation": encode_board(self.board, self.current_mark, self.rows, self.columns),
            "action_mask": valid_action_mask(self.board, self.rows, self.columns),
        }

    def _info(self, **extra: Any) -> dict[str, Any]:
        info = {
            "board": list(self.board),
            "current_mark": self.current_mark,
            "winner": self.winner,
            "move_count": self.move_count,
            "legal_actions": self.available_actions(),
        }
        info.update(extra)
        return info
