import numpy as np

from connectx.agents.q_learning import TabularQAgent, canonical_state
from connectx.envs.connectx_env import ConnectXConfig


def test_canonical_state_uses_current_player_perspective():
    board = [0] * 20
    board[19] = 2
    board[18] = 1

    state = canonical_state(board, mark=2)

    assert state[19] == 1
    assert state[18] == 2


def test_tabular_q_agent_masks_full_columns():
    config = ConnectXConfig(rows=4, columns=5, inarow=3)
    agent = TabularQAgent(config=config)
    board = [0] * 20
    for row in range(4):
        board[row * 5 + 0] = 1 if row % 2 == 0 else 2
    state = canonical_state(board, 1)
    agent.q_table[state] = np.asarray([10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    assert agent.act(board, mark=1, epsilon=0.0) != 0
