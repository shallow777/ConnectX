import numpy as np
import pytest

from connectx.envs.connectx_env import (
    ConnectXEnv,
    check_winner,
    encode_board,
    valid_action_mask,
)


def test_encode_board_uses_current_player_perspective():
    board = [0] * 42
    board[5 * 7 + 0] = 2
    board[5 * 7 + 1] = 1

    encoded = encode_board(board, current_mark=2, rows=6, columns=7)

    assert encoded.shape == (2, 6, 7)
    assert encoded.dtype == np.float32
    assert encoded[0, 5, 0] == 1.0
    assert encoded[1, 5, 1] == 1.0
    assert encoded[0, 5, 1] == 0.0
    assert encoded[1, 5, 0] == 0.0


def test_env_step_flips_observation_to_next_player_view():
    env = ConnectXEnv()
    obs, info = env.reset()

    assert obs["observation"].shape == (2, 6, 7)
    assert obs["action_mask"].tolist() == [1] * 7
    assert info["current_mark"] == 1

    obs, reward, terminated, truncated, info = env.step(0)

    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["current_mark"] == 2
    assert obs["observation"][0].sum() == 0.0
    assert obs["observation"][1, 5, 0] == 1.0


def test_valid_action_mask_blocks_full_columns():
    board = [0] * 42
    for row in range(6):
        board[row * 7 + 3] = 1 if row % 2 == 0 else 2

    mask = valid_action_mask(board, rows=6, columns=7)

    assert mask.dtype == np.int8
    assert mask.tolist() == [1, 1, 1, 0, 1, 1, 1]


@pytest.mark.parametrize(
    "positions",
    [
        [5 * 7 + c for c in range(4)],
        [r * 7 + 0 for r in range(2, 6)],
        [2 * 7 + 0, 3 * 7 + 1, 4 * 7 + 2, 5 * 7 + 3],
        [5 * 7 + 0, 4 * 7 + 1, 3 * 7 + 2, 2 * 7 + 3],
    ],
)
def test_check_winner_detects_all_four_directions(positions):
    board = [0] * 42
    for idx in positions:
        board[idx] = 1

    assert check_winner(board, mark=1, rows=6, columns=7, inarow=4)
    assert not check_winner(board, mark=2, rows=6, columns=7, inarow=4)


def test_illegal_action_terminates_with_loss():
    env = ConnectXEnv()
    env.reset()
    for row in range(6):
        env.board[row * 7] = 1 if row % 2 == 0 else 2

    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert reward == -1.0
    assert info["illegal_action"] is True
    assert obs["action_mask"][0] == 0
