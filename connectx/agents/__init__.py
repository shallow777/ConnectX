"""Agent implementations live here."""

from connectx.agents.lookahead import (
    immediate_winning_actions,
    safe_policy_action,
    tactical_action,
    tactical_agent,
    wrap_with_tactical_safety,
)
from connectx.agents.q_learning import TabularQAgent

__all__ = [
    "immediate_winning_actions",
    "safe_policy_action",
    "tactical_action",
    "tactical_agent",
    "wrap_with_tactical_safety",
    "TabularQAgent",
]
