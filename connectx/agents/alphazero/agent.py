"""把训练好的 AlphaZero 模型包装成 (obs, config) -> action 的推理 agent.

实战推理: MCTS 关闭 Dirichlet 噪声、温度趋近 0 (取访问次数最多的列),
外层再套战术安全层兜底。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from connectx.agents.alphazero.mcts import AlphaZeroMCTS, MCTSConfig, TorchEvaluator
from connectx.agents.alphazero.network import AlphaZeroNet, load_checkpoint
from connectx.agents.lookahead import safe_policy_action
from connectx.agents.utils import normalize_config, obs_board_mark
from connectx.envs.connectx_env import ConnectXConfig


def make_alphazero_agent_from_model(
    model: AlphaZeroNet,
    *,
    simulations: int = 100,
    device: str = "cpu",
    tactical_safety: bool = True,
) -> Callable[[dict[str, Any], ConnectXConfig], int]:
    """从已加载的模型构造 agent (评估模式: 无噪声, 温度 ~0 即贪心选列)."""

    def agent(obs: dict[str, Any], config: ConnectXConfig) -> int:
        board, mark = obs_board_mark(obs)
        normalized = normalize_config(config)
        evaluator = TorchEvaluator(model, normalized, device=device)
        mcts = AlphaZeroMCTS(evaluator, normalized, MCTSConfig(simulations=simulations))
        policy = mcts.search(board, mark, add_noise=False, temperature=1e-6)
        action = int(np.argmax(policy))
        if tactical_safety:
            action = safe_policy_action(board, mark, normalized, action)
        return action

    return agent


def make_alphazero_agent(
    checkpoint_path: str | Path,
    *,
    simulations: int = 100,
    device: str = "cpu",
    tactical_safety: bool = True,
) -> Callable[[dict[str, Any], ConnectXConfig], int]:
    """从 checkpoint 路径构造 agent; 模型首次调用时才加载 (lazy load)."""

    def agent(obs: dict[str, Any], config: ConnectXConfig) -> int:
        if not hasattr(agent, "_model"):
            model, _payload = load_checkpoint(checkpoint_path, map_location=device)
            agent._model = model  # type: ignore[attr-defined]
        model = agent._model  # type: ignore[attr-defined]
        return make_alphazero_agent_from_model(
            model,
            simulations=simulations,
            device=device,
            tactical_safety=tactical_safety,
        )(obs, config)

    return agent
