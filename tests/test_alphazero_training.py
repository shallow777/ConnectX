import numpy as np
import pytest

from connectx.agents.alphazero.replay_buffer import ReplayBuffer
from connectx.envs.connectx_env import mirror_encoded_state, mirror_mask, mirror_policy
from connectx.training.train_alphazero import learning_rate_for_generation


def test_mirror_augmentation_reverses_columns():
    state = np.arange(84, dtype=np.float32).reshape(2, 6, 7)
    policy = np.arange(7, dtype=np.float32)
    mask = np.array([1, 0, 1, 0, 1, 0, 1], dtype=bool)

    mirrored_state = mirror_encoded_state(state, 7)
    mirrored_policy = mirror_policy(policy, 7)
    mirrored_mask = mirror_mask(mask, 7)

    assert np.array_equal(mirrored_state[:, :, 0], state[:, :, 6])
    assert mirrored_policy[0] == policy[6]
    assert mirrored_mask[0] == mask[6]


def test_replay_buffer_prunes_old_generations():
    buffer = ReplayBuffer(capacity=1000, max_generations=2)
    sample = (
        np.zeros((2, 6, 7), dtype=np.float32),
        np.ones(7, dtype=np.float32),
        0.5,
        np.ones(7, dtype=bool),
    )
    buffer.add_game([sample], generation=0)
    buffer.add_game([sample], generation=1)
    buffer.add_game([sample], generation=2)
    assert len(buffer) == 2
    assert min(buffer._generations) == 1


def test_learning_rate_decays_in_later_generations():
    lr_start = learning_rate_for_generation(
        10,
        base_lr=1e-3,
        final_lr=2e-4,
        total_generations=100,
        decay_start_generation=20,
    )
    lr_mid = learning_rate_for_generation(
        60,
        base_lr=1e-3,
        final_lr=2e-4,
        total_generations=100,
        decay_start_generation=20,
    )
    lr_end = learning_rate_for_generation(
        100,
        base_lr=1e-3,
        final_lr=2e-4,
        total_generations=100,
        decay_start_generation=20,
    )
    assert lr_start == 1e-3
    assert lr_end == pytest.approx(2e-4)
    assert lr_mid < lr_start
    assert lr_mid > lr_end
