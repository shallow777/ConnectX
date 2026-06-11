import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_submission(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ppo_submission_agent_returns_legal_column(tmp_path):
    """端到端验证: 导出 PPO 权重 -> 渲染单文件 submission -> agent 返回合法列。

    需要本地有训练好的 PPO checkpoint, 没有则跳过 (skip if unavailable)。
    """
    from connectx.submission.make_submission import export_ppo_model, render_submission

    candidates = sorted(Path("runs/ppo/checkpoints").glob("ppo_*.zip"))
    if not candidates and Path("runs/ppo/best_model/best_model.zip").exists():
        candidates = [Path("runs/ppo/best_model/best_model.zip")]
    if not candidates:
        pytest.skip("no local PPO checkpoint available")

    weights_b64, _meta = export_ppo_model(candidates[-1])
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
