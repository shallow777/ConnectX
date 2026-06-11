from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

import numpy as np

from connectx.envs.connectx_env import ConnectXConfig, ConnectXEnv


AgentFn = Callable[[dict[str, Any], ConnectXConfig], int]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    agent: AgentFn


@dataclass(frozen=True)
class GameResult:
    winner: int
    first_player: int
    moves: list[int]
    final_board: list[int]
    illegal_action: bool = False
    truncated: bool = False


@dataclass
class MatchupStats:
    agent_a: str
    agent_b: str
    games: int = 0
    wins: dict[str, int] = field(default_factory=dict)
    draws: int = 0
    illegal_losses: dict[str, int] = field(default_factory=dict)
    first_player_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (self.agent_a, self.agent_b):
            self.wins.setdefault(name, 0)
            self.illegal_losses.setdefault(name, 0)
            self.first_player_counts.setdefault(name, 0)

    def win_rate(self, name: str) -> float:
        if self.games == 0:
            return 0.0
        return self.wins[name] / self.games

    def draw_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.draws / self.games

    def record(self, result: GameResult, first_name: str, second_name: str) -> None:
        self.games += 1
        self.first_player_counts[first_name] += 1

        if result.winner == 0:
            self.draws += 1
            return

        winner_name = first_name if result.winner == 1 else second_name
        loser_name = second_name if result.winner == 1 else first_name
        self.wins[winner_name] += 1
        if result.illegal_action:
            self.illegal_losses[loser_name] += 1


@dataclass
class ArenaStandings:
    agents: tuple[str, ...]
    matchups: dict[tuple[str, str], MatchupStats]

    def matchup(self, agent_a: str, agent_b: str) -> MatchupStats:
        key = (agent_a, agent_b)
        if key in self.matchups:
            return self.matchups[key]
        reverse_key = (agent_b, agent_a)
        if reverse_key in self.matchups:
            return self.matchups[reverse_key]
        raise KeyError(f"No matchup recorded for {agent_a!r} vs {agent_b!r}")

    def win_rate(self, agent: str, opponent: str) -> float:
        return self.matchup(agent, opponent).win_rate(agent)

    def total_score(self, agent: str) -> float:
        score = 0.0
        for stats in self.matchups.values():
            if agent not in stats.wins:
                continue
            score += stats.wins[agent] + 0.5 * stats.draws
        return score

    def win_rate_matrix(self) -> dict[str, dict[str, float | None]]:
        matrix: dict[str, dict[str, float | None]] = {}
        for agent in self.agents:
            row: dict[str, float | None] = {}
            for opponent in self.agents:
                row[opponent] = None if agent == opponent else self.win_rate(agent, opponent)
            matrix[agent] = row
        return matrix

    def summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (agent_a, agent_b), stats in self.matchups.items():
            rows.append(
                {
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "games": stats.games,
                    "agent_a_win_rate": stats.win_rate(agent_a),
                    "agent_b_win_rate": stats.win_rate(agent_b),
                    "draw_rate": stats.draw_rate(),
                    "agent_a_wins": stats.wins[agent_a],
                    "agent_b_wins": stats.wins[agent_b],
                    "draws": stats.draws,
                }
            )
        return rows

    def format_win_rate_table(self) -> str:
        names = list(self.agents)
        width = max(8, *(len(name) for name in names)) + 2
        header = " " * width + "".join(name.rjust(width) for name in names)
        lines = [header]
        matrix = self.win_rate_matrix()
        for agent in names:
            cells = []
            for opponent in names:
                value = matrix[agent][opponent]
                cells.append("-".rjust(width) if value is None else f"{value:.3f}".rjust(width))
            lines.append(agent.rjust(width) + "".join(cells))
        return "\n".join(lines)


def play_game(
    first_agent: AgentFn,
    second_agent: AgentFn,
    *,
    rows: int = 6,
    columns: int = 7,
    inarow: int = 4,
    max_moves: int | None = None,
) -> GameResult:
    env = ConnectXEnv(rows=rows, columns=columns, inarow=inarow)
    obs, info = env.reset()
    agents = {1: first_agent, 2: second_agent}
    moves: list[int] = []
    terminated = False
    truncated = False
    illegal_action = False
    max_moves = max_moves or rows * columns

    while not terminated and not truncated:
        if len(moves) >= max_moves:
            truncated = True
            break

        agent = agents[env.current_mark]
        action = agent(_agent_observation(obs, info), env.config)
        obs, _reward, terminated, truncated, info = env.step(int(action))
        moves.append(int(action))
        illegal_action = bool(info.get("illegal_action", False))

    return GameResult(
        winner=env.winner,
        first_player=1,
        moves=moves,
        final_board=list(env.board),
        illegal_action=illegal_action,
        truncated=truncated,
    )


def evaluate_pair(
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    *,
    games: int = 200,
    rows: int = 6,
    columns: int = 7,
    inarow: int = 4,
) -> MatchupStats:
    if games <= 0:
        raise ValueError("games must be positive")

    stats = MatchupStats(agent_a.name, agent_b.name)
    for game_idx in range(games):
        # 轮流执先手, 消除 ConnectX 先手优势对胜率统计的影响
        if game_idx % 2 == 0:
            first, second = agent_a, agent_b
        else:
            first, second = agent_b, agent_a

        result = play_game(
            first.agent,
            second.agent,
            rows=rows,
            columns=columns,
            inarow=inarow,
        )
        stats.record(result, first.name, second.name)

    return stats


def evaluate_agents(
    agents: list[AgentSpec] | tuple[AgentSpec, ...],
    *,
    games_per_pair: int = 200,
    rows: int = 6,
    columns: int = 7,
    inarow: int = 4,
) -> ArenaStandings:
    if len(agents) < 2:
        raise ValueError("At least two agents are required")
    if games_per_pair <= 0:
        raise ValueError("games_per_pair must be positive")

    matchups: dict[tuple[str, str], MatchupStats] = {}
    for agent_a, agent_b in combinations(agents, 2):
        stats = evaluate_pair(
            agent_a,
            agent_b,
            games=games_per_pair,
            rows=rows,
            columns=columns,
            inarow=inarow,
        )
        matchups[(agent_a.name, agent_b.name)] = stats

    return ArenaStandings(tuple(agent.name for agent in agents), matchups)


def random_agent(obs: dict[str, Any], config: ConnectXConfig) -> int:
    legal = np.flatnonzero(obs["action_mask"])
    if legal.size == 0:
        return 0
    return int(np.random.choice(legal))


def first_legal_agent(obs: dict[str, Any], config: ConnectXConfig) -> int:
    del config
    legal = np.flatnonzero(obs["action_mask"])
    return int(legal[0]) if legal.size else 0


def _agent_observation(obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
    prepared: dict[str, Any] = {
        "observation": obs["observation"].copy(),
        "action_mask": obs["action_mask"].copy(),
        "board": list(info["board"]),
        "mark": int(info["current_mark"]),
    }
    return prepared
