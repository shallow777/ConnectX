# Solver v1 Diagnosis (expanded replay)

## Per-step logging (sample opening game)

- ply 0 mark=1: solver=True conclusion=win solver_s=0.022 mcts_s=0.000 sims=0 move=3
- ply 1 mark=2: solver=True conclusion=loss solver_s=0.012 mcts_s=1.750 sims=700 move=3
- ply 2 mark=1: solver=True conclusion=win solver_s=0.012 mcts_s=0.000 sims=0 move=0
- ply 3 mark=2: solver=True conclusion=loss solver_s=0.006 mcts_s=1.756 sims=683 move=2
- ply 4 mark=1: solver=True conclusion=win solver_s=0.001 mcts_s=0.000 sims=0 move=1
- ply 5 mark=2: solver=True conclusion=win solver_s=0.000 mcts_s=0.000 sims=0 move=2

## First divergence vs rollback

Game 0 ply 1: v1 played col 0 (optimal), rollback played col 2 (optimal).
Offline verdict: **unclear branch chose worse move** at divergence.

## Trajectory diversity (40 games)

Distinct full trajectories: **2** / 40 games.

High repetition confirms deterministic play when solver bypasses MCTS.

## v1 root causes (summary)

1. Solver bypass on draw/win proof skips MCTS entirely.
2. 50% budget spent on solver before MCTS starts.
3. No empty-cell gate — solver runs from move 1.

## v2/hybrid fix

- Trigger only when empty ≤ 18; adaptive backoff on timeout.
- Solver ≤ 30% budget; timeout returns unknown (no partial results).
- TB-style root filter: win → win moves, draw-only → draws, all-loss → no filter.
- Opening hardcode: empty → col 3; one piece → adjacent.
- Base: gen8 cached submission with forward cache.
