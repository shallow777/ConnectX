from connectx.agents.lookahead import safe_policy_action, tactical_action
from connectx.envs.connectx_env import ConnectXConfig


def test_tactical_action_takes_immediate_win():
    config = ConnectXConfig()
    board = [0] * 42
    for col in range(3):
        board[5 * 7 + col] = 1

    assert tactical_action(board, 1, config) == 3


def test_tactical_action_blocks_opponent_immediate_win():
    config = ConnectXConfig()
    board = [0] * 42
    for col in range(3):
        board[5 * 7 + col] = 2

    assert tactical_action(board, 1, config) == 3


def test_tactical_action_blocks_open_two_before_fork():
    """Opponent bottom row at cols 3,4: must block extension, not stack center."""
    config = ConnectXConfig()
    board = [0] * 42
    board[5 * 7 + 3] = 2
    board[5 * 7 + 4] = 2

    action = tactical_action(board, 1, config)
    assert action in (2, 5)


def test_safe_policy_falls_back_when_policy_action_is_illegal():
    config = ConnectXConfig()
    board = [0] * 42
    for row in range(6):
        board[row * 7] = 1 if row % 2 == 0 else 2

    assert safe_policy_action(board, 1, config, policy_action=0) != 0
