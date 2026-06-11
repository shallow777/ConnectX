from connectx.evaluation.arena import AgentSpec, evaluate_agents, play_game


def center_agent(obs, config):
    mask = obs["action_mask"]
    center = config.columns // 2
    if mask[center]:
        return center
    return next(idx for idx, valid in enumerate(mask) if valid)


def left_agent(obs, config):
    return next(idx for idx, valid in enumerate(obs["action_mask"]) if valid)


def test_play_game_reports_winner_and_move_history():
    result = play_game(center_agent, left_agent)

    assert result.winner in {0, 1, 2}
    assert result.first_player == 1
    assert result.moves
    assert all(0 <= move < 7 for move in result.moves)


def test_evaluate_agents_alternates_first_player_and_counts_results():
    standings = evaluate_agents(
        [
            AgentSpec("center", center_agent),
            AgentSpec("left", left_agent),
        ],
        games_per_pair=4,
    )

    matchup = standings.matchups[("center", "left")]

    assert matchup.games == 4
    assert matchup.first_player_counts == {"center": 2, "left": 2}
    assert matchup.wins["center"] + matchup.wins["left"] + matchup.draws == 4
    assert standings.win_rate("center", "left") == matchup.wins["center"] / 4
