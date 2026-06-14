from __future__ import annotations

from connectx.agents.reward_shaping import RewardShapingConfig, compute_step_reward
from connectx.envs.connectx_env import ConnectXConfig, next_board


def test_terminal_win_reward():
    config = ConnectXConfig(rows=4, columns=5, inarow=3)
    shaping = RewardShapingConfig()
    before = [0] * 20
    # Column 0: three marks stacked with room for a fourth on top in row 0.
    before[0] = before[5] = before[10] = 1
    after = next_board(before, 0, 1, config.rows, config.columns)
    reward = compute_step_reward(before, after, 1, config, shaping)
    assert reward == 1.0


def test_shaping_nonzero_on_progress():
    config = ConnectXConfig(rows=6, columns=7, inarow=4)
    shaping = RewardShapingConfig()
    before = [0] * 42
    after = next_board(before, 3, 1, config.rows, config.columns)
    reward = compute_step_reward(before, after, 1, config, shaping)
    assert reward > 0.0


def test_sparse_when_disabled():
    config = ConnectXConfig(rows=6, columns=7, inarow=4)
    shaping = RewardShapingConfig(enabled=False)
    before = [0] * 42
    after = next_board(before, 3, 1, config.rows, config.columns)
    reward = compute_step_reward(before, after, 1, config, shaping)
    assert reward == 0.0


def test_alphazero_sparse_targets_match_terminal_outcome():
    from connectx.agents.reward_shaping import compute_alphazero_value_targets

    marks = [1, 2, 1]
    targets = compute_alphazero_value_targets(marks, winner=1, shaping_rewards=[0.0, 0.0, 0.0])
    assert targets == [1.0, -1.0, 1.0]


def test_alphazero_shaped_targets_differ_from_sparse():
    from connectx.agents.reward_shaping import compute_alphazero_value_targets

    marks = [1, 2]
    sparse = compute_alphazero_value_targets(marks, winner=1, shaping_rewards=[0.0, 0.0])
    shaped = compute_alphazero_value_targets(marks, winner=1, shaping_rewards=[0.0, 0.05])
    assert shaped[0] != sparse[0]
