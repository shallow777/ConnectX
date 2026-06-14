import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_submission(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_folds_batchnorm():
    from connectx.submission.make_submission import export_alphazero_checkpoint

    checkpoint = Path("runs/alphazero_push/checkpoints/generation_0008_accepted.pt")
    if not checkpoint.exists():
        pytest.skip("push checkpoint not available")

    weights_b64, _config = export_alphazero_checkpoint(checkpoint)
    import base64
    import io
    import json
    import numpy as np

    data = np.load(io.BytesIO(base64.b64decode(weights_b64)), allow_pickle=False)
    meta = json.loads(str(data["__meta__"][0]))
    assert meta.get("folded_bn") is True
    assert "stem.1.weight" not in data.files
    assert "stem.0.bias" in data.files


def test_alphazero_opening_plays_center_first_move(tmp_path):
    from connectx.submission.make_submission import export_alphazero_checkpoint, render_submission

    checkpoint = Path("runs/alphazero_push/checkpoints/generation_0008_accepted.pt")
    if not checkpoint.exists():
        pytest.skip("push checkpoint not available")

    weights_b64, _meta = export_alphazero_checkpoint(checkpoint)
    submission_path = tmp_path / "submission.py"
    submission_path.write_text(render_submission("alphazero", weights_b64))
    module = _load_submission(submission_path)

    action = module.agent(
        SimpleNamespace(board=[0] * 42, mark=1, remainingOverageTime=60),
        SimpleNamespace(rows=6, columns=7, inarow=4, actTimeout=2, timeout=2),
    )
    assert action == 3


def test_alphazero_time_budget_uses_overage_in_opening(tmp_path):
    from connectx.submission.make_submission import export_alphazero_checkpoint, render_submission

    checkpoint = Path("runs/alphazero_push/checkpoints/generation_0008_accepted.pt")
    if not checkpoint.exists():
        pytest.skip("push checkpoint not available")

    weights_b64, _meta = export_alphazero_checkpoint(checkpoint)
    submission_path = tmp_path / "submission.py"
    submission_path.write_text(render_submission("alphazero", weights_b64))
    module = _load_submission(submission_path)

    board = [0] * 42
    board[5 * 7 + 3] = 1
    board[4 * 7 + 3] = 2
    budget = module._time_budget(
        SimpleNamespace(board=board, mark=1, remainingOverageTime=60),
        SimpleNamespace(rows=6, columns=7, inarow=4, actTimeout=2, timeout=2),
        board,
        6,
        7,
    )
    assert budget >= 3.3


def test_alphazero_blocks_open_two_fork(tmp_path):
    from connectx.submission.make_submission import export_alphazero_checkpoint, render_submission

    checkpoint = Path("runs/alphazero_push/checkpoints/generation_0008_accepted.pt")
    if not checkpoint.exists():
        pytest.skip("push checkpoint not available")

    weights_b64, _meta = export_alphazero_checkpoint(checkpoint)
    submission_path = tmp_path / "submission.py"
    submission_path.write_text(render_submission("alphazero", weights_b64))
    module = _load_submission(submission_path)

    board = [0] * 42
    board[5 * 7 + 3] = 2
    board[5 * 7 + 4] = 2

    action = module.agent(
        SimpleNamespace(board=board, mark=1, remainingOverageTime=60),
        SimpleNamespace(rows=6, columns=7, inarow=4, actTimeout=2, timeout=2),
    )
    assert action in (2, 5)


def test_alphazero_blocks_opponent_horizontal_four(tmp_path):
    from connectx.submission.make_submission import export_alphazero_checkpoint, render_submission

    checkpoint = Path("runs/alphazero_push/checkpoints/generation_0008_accepted.pt")
    if not checkpoint.exists():
        pytest.skip("push checkpoint not available")

    weights_b64, _meta = export_alphazero_checkpoint(checkpoint)
    submission_path = tmp_path / "submission.py"
    submission_path.write_text(render_submission("alphazero", weights_b64))
    module = _load_submission(submission_path)

    board = [0] * 42
    for col in (1, 2, 3):
        board[5 * 7 + col] = 2
    board[5 * 7 + 0] = 1

    action = module.agent(
        SimpleNamespace(board=board, mark=1, remainingOverageTime=60),
        SimpleNamespace(rows=6, columns=7, inarow=4, actTimeout=2, timeout=2),
    )
    assert action == 4


def test_ppo_submission_agent_returns_legal_column(tmp_path):
    from connectx.submission.make_submission import export_ppo_model, render_submission

    smoke_model = Path("runs/ppo/smoke/checkpoints/ppo_2048.zip")
    if not smoke_model.exists():
        pytest.skip("smoke PPO checkpoint not available")

    weights_b64, _meta = export_ppo_model(smoke_model)
    submission_path = tmp_path / "submission.py"
    submission_path.write_text(render_submission("ppo", weights_b64))
    module = _load_submission(submission_path)

    board = [0] * 42
    for row in range(6):
        board[row * 7 + 3] = 1

    action = module.agent(
        SimpleNamespace(board=board, mark=2, remainingOverageTime=60),
        SimpleNamespace(rows=6, columns=7, inarow=4, actTimeout=2, timeout=2),
    )
    assert action in [0, 1, 2, 4, 5, 6]
