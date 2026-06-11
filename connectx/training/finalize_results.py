from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from connectx.agents.alphazero.agent import make_alphazero_agent
from connectx.agents.dqn import make_dqn_agent
from connectx.agents.lookahead import wrap_with_tactical_safety
from connectx.agents.ppo_selfplay import make_sb3_ppo_agent
from connectx.agents.q_learning import TabularQAgent
from connectx.evaluation.arena import AgentSpec, evaluate_agents
from connectx.evaluation.kaggle_eval import evaluate_against_negamax
from connectx.submission.make_submission import (
    export_alphazero_checkpoint,
    export_ppo_model,
    render_submission,
    select_best_kind,
    validate_submission,
)


def plot_csv(path: Path, output: Path, *, x: str, y: str, title: str) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty or y not in frame.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=frame, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def latest_checkpoint(checkpoint_dir: Path, pattern: str) -> Path | None:
    paths = sorted(checkpoint_dir.glob(pattern))
    return paths[-1] if paths else None


def best_alphazero_from_runs() -> Path | None:
    best_path: Path | None = None
    best_rate = -1.0
    for run_dir in (Path("runs/alphazero"), Path("runs/alphazero_overnight")):
        curve_path = run_dir / "negamax_curve.csv"
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        final = ckpt_dir / "alphazero_final.pt"
        if curve_path.exists():
            frame = pd.read_csv(curve_path)
            if not frame.empty and "negamax_win_rate" in frame.columns:
                row = frame.loc[frame["negamax_win_rate"].idxmax()]
                generation = int(row["generation"])
                accepted = ckpt_dir / f"generation_{generation:04d}_accepted.pt"
                candidate = accepted if accepted.exists() else ckpt_dir / f"generation_{generation:04d}_champion.pt"
                rate = float(row["negamax_win_rate"])
                if rate > best_rate and candidate.exists():
                    best_rate = rate
                    best_path = candidate
        if final.exists() and best_path is None:
            best_path = final
    return best_path


def best_ppo_checkpoint() -> Path | None:
    return latest_checkpoint(Path("runs/ppo/checkpoints"), "ppo_*.zip")


def copy_curve(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def negamax_row(algorithm: str, result, *, board: str, checkpoint: str) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "board": board,
        "checkpoint": checkpoint,
        "negamax_games": result.games,
        "negamax_wins": result.wins,
        "negamax_losses": result.losses,
        "negamax_draws": result.draws,
        "negamax_win_rate": result.win_rate,
        "negamax_mean_reward": result.mean_reward,
    }


def plot_negamax_comparison(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    plt.figure(figsize=(9, 4))
    sns.barplot(data=frame, x="algorithm", y="negamax_win_rate", hue="board", dodge=False)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Negamax win rate")
    plt.title("Final agent strength vs Negamax")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_training_curves(results_dir: Path) -> None:
    curves = results_dir / "curves"
    plot_csv(curves / "ppo_negamax_curve.csv", results_dir / "ppo_negamax_curve.png", x="timesteps", y="negamax_win_rate", title="PPO vs Negamax")
    plot_csv(curves / "alphazero_negamax_curve.csv", results_dir / "alphazero_negamax_curve.png", x="generation", y="negamax_win_rate", title="AlphaZero (main) vs Negamax")
    if (curves / "alphazero_overnight_negamax_curve.csv").exists():
        plot_csv(
            curves / "alphazero_overnight_negamax_curve.csv",
            results_dir / "alphazero_overnight_negamax_curve.png",
            x="generation",
            y="negamax_win_rate",
            title="AlphaZero (overnight) vs Negamax",
        )
    plot_csv(curves / "dqn_learning_curve.csv", results_dir / "dqn_learning_curve.png", x="episode", y="mean_loss", title="DQN mean loss")
    plot_csv(curves / "q_learning_learning_curve.csv", results_dir / "q_learning_curve.png", x="episode", y="q_states", title="Q-learning table size")

    dqn_path = curves / "dqn_learning_curve.csv"
    if dqn_path.exists():
        frame = pd.read_csv(dqn_path)
        if "winner" in frame.columns:
            frame["learner_win"] = (frame["winner"] == 1.0).astype(float)
            frame["rolling_win_rate"] = frame["learner_win"].rolling(500, min_periods=50).mean()
            plt.figure(figsize=(8, 4))
            sns.lineplot(data=frame, x="episode", y="rolling_win_rate")
            plt.title("DQN self-play rolling win rate")
            plt.tight_layout()
            plt.savefig(results_dir / "dqn_selfplay_win_rate.png", dpi=150)
            plt.close()


def finalize(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    curves_dir = results_dir / "curves"
    results_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    ppo_ckpt = Path(args.ppo_model) if args.ppo_model else best_ppo_checkpoint()
    az_ckpt = Path(args.alphazero_checkpoint) if args.alphazero_checkpoint else best_alphazero_from_runs()
    az_main_ckpt = Path("runs/alphazero/checkpoints/alphazero_final.pt")
    dqn_ckpt = Path(args.dqn_model) if args.dqn_model else Path("runs/dqn/dqn.pt")
    ql_ckpt = Path("runs/q_learning/q_learning.pkl")

    # Preserve raw learning curves for all four algorithms.
    copy_curve(Path("runs/q_learning/learning_curve.csv"), curves_dir / "q_learning_learning_curve.csv")
    copy_curve(Path("runs/dqn/learning_curve.csv"), curves_dir / "dqn_learning_curve.csv")
    copy_curve(Path("runs/ppo/negamax_curve.csv"), curves_dir / "ppo_negamax_curve.csv")
    copy_curve(Path("runs/alphazero/negamax_curve.csv"), curves_dir / "alphazero_negamax_curve.csv")
    copy_curve(Path("runs/alphazero_overnight/negamax_curve.csv"), curves_dir / "alphazero_overnight_negamax_curve.csv")

    plot_training_curves(results_dir)

    # Final negamax benchmarks (standard 6x7 + small-board Q-learning baseline).
    negamax_rows: list[dict[str, Any]] = []
    if ppo_ckpt and ppo_ckpt.exists():
        negamax_rows.append(
            negamax_row(
                "ppo",
                evaluate_against_negamax(
                    wrap_with_tactical_safety(make_sb3_ppo_agent(ppo_ckpt)),
                    games=args.negamax_games,
                ),
                board="6x7 connect-4",
                checkpoint=str(ppo_ckpt),
            )
        )
    if dqn_ckpt.exists():
        negamax_rows.append(
            negamax_row(
                "dqn",
                evaluate_against_negamax(make_dqn_agent(dqn_ckpt, device="cpu"), games=args.negamax_games),
                board="6x7 connect-4",
                checkpoint=str(dqn_ckpt),
            )
        )
    if az_main_ckpt.exists():
        negamax_rows.append(
            negamax_row(
                "alphazero_main",
                evaluate_against_negamax(
                    make_alphazero_agent(az_main_ckpt, simulations=args.alphazero_simulations, device="cpu", tactical_safety=True),
                    games=args.negamax_games,
                ),
                board="6x7 connect-4",
                checkpoint=str(az_main_ckpt),
            )
        )
    if az_ckpt and az_ckpt.exists() and str(az_ckpt) != str(az_main_ckpt):
        negamax_rows.append(
            negamax_row(
                "alphazero_best",
                evaluate_against_negamax(
                    make_alphazero_agent(az_ckpt, simulations=args.alphazero_simulations, device="cpu", tactical_safety=True),
                    games=args.negamax_games,
                ),
                board="6x7 connect-4",
                checkpoint=str(az_ckpt),
            )
        )
    if ql_ckpt.exists():
        ql = TabularQAgent.load(ql_ckpt)
        negamax_rows.append(
            negamax_row(
                "q_learning",
                evaluate_against_negamax(
                    ql.agent_fn(tactical_safety=True),
                    games=args.negamax_games,
                    rows=ql.config.rows,
                    columns=ql.config.columns,
                    inarow=ql.config.inarow,
                ),
                board=f"{ql.config.rows}x{ql.config.columns} connect-{ql.config.inarow}",
                checkpoint=str(ql_ckpt),
            )
        )

    with (results_dir / "negamax_final_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(negamax_rows[0].keys()) if negamax_rows else [])
        if negamax_rows:
            writer.writeheader()
            writer.writerows(negamax_rows)
    plot_negamax_comparison(negamax_rows, results_dir / "negamax_final_comparison.png")

    summary_rows = []
    for row in negamax_rows:
        summary_rows.append({**row, "metric": "negamax_win_rate", "metric_value": row["negamax_win_rate"]})
    if ppo_ckpt and (curves_dir / "ppo_negamax_curve.csv").exists():
        ppo_frame = pd.read_csv(curves_dir / "ppo_negamax_curve.csv")
        summary_rows.append(
            {
                "algorithm": "ppo",
                "board": "6x7 connect-4",
                "checkpoint": str(ppo_ckpt),
                "metric": "peak_negamax_win_rate",
                "metric_value": float(ppo_frame["negamax_win_rate"].max()),
            }
        )
    if (curves_dir / "alphazero_negamax_curve.csv").exists():
        az_frame = pd.read_csv(curves_dir / "alphazero_negamax_curve.csv")
        summary_rows.append(
            {
                "algorithm": "alphazero_main",
                "board": "6x7 connect-4",
                "checkpoint": str(az_main_ckpt),
                "metric": "peak_negamax_win_rate",
                "metric_value": float(az_frame["negamax_win_rate"].max()),
            }
        )
    if (curves_dir / "alphazero_overnight_negamax_curve.csv").exists():
        azo_frame = pd.read_csv(curves_dir / "alphazero_overnight_negamax_curve.csv")
        summary_rows.append(
            {
                "algorithm": "alphazero_overnight",
                "board": "6x7 connect-4",
                "checkpoint": str(az_ckpt) if az_ckpt else "",
                "metric": "peak_negamax_win_rate",
                "metric_value": float(azo_frame["negamax_win_rate"].max()),
            }
        )
    if (curves_dir / "q_learning_learning_curve.csv").exists():
        ql_frame = pd.read_csv(curves_dir / "q_learning_learning_curve.csv")
        summary_rows.append(
            {
                "algorithm": "q_learning",
                "board": "4x5 connect-3",
                "checkpoint": str(ql_ckpt),
                "metric": "final_q_table_states",
                "metric_value": float(ql_frame["q_states"].iloc[-1]),
            }
        )
    if (curves_dir / "dqn_learning_curve.csv").exists():
        dqn_frame = pd.read_csv(curves_dir / "dqn_learning_curve.csv")
        summary_rows.append(
            {
                "algorithm": "dqn",
                "board": "6x7 connect-4",
                "checkpoint": str(dqn_ckpt),
                "metric": "final_mean_loss",
                "metric_value": float(dqn_frame["mean_loss"].iloc[-1]),
            }
        )

    with (results_dir / "algorithms_summary.csv").open("w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    (results_dir / "algorithms_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")

    manifest = {
        "q_learning": {"checkpoint": str(ql_ckpt), "curve": str(curves_dir / "q_learning_learning_curve.csv"), "board": "4x5 connect-3"},
        "dqn": {"checkpoint": str(dqn_ckpt), "curve": str(curves_dir / "dqn_learning_curve.csv"), "board": "6x7 connect-4"},
        "ppo": {"checkpoint": str(ppo_ckpt) if ppo_ckpt else "", "curve": str(curves_dir / "ppo_negamax_curve.csv"), "board": "6x7 connect-4"},
        "alphazero_main": {"checkpoint": str(az_main_ckpt), "curve": str(curves_dir / "alphazero_negamax_curve.csv"), "board": "6x7 connect-4"},
        "alphazero_overnight": {
            "checkpoint": str(az_ckpt) if az_ckpt else "",
            "curve": str(curves_dir / "alphazero_overnight_negamax_curve.csv"),
            "board": "6x7 connect-4",
        },
    }
    (results_dir / "algorithms_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    agents: list[AgentSpec] = []
    if ppo_ckpt and ppo_ckpt.exists():
        agents.append(AgentSpec("ppo", wrap_with_tactical_safety(make_sb3_ppo_agent(ppo_ckpt))))
    if az_ckpt and az_ckpt.exists():
        agents.append(
            AgentSpec(
                "alphazero",
                make_alphazero_agent(az_ckpt, simulations=args.alphazero_simulations, device="cpu", tactical_safety=True),
            )
        )
    if dqn_ckpt.exists():
        agents.append(AgentSpec("dqn", make_dqn_agent(dqn_ckpt, device="cpu")))

    if len(agents) < 2:
        raise RuntimeError("Need at least two trained agents for arena evaluation.")

    standings = evaluate_agents(agents, games_per_pair=args.arena_games)
    matrix = standings.win_rate_matrix()
    table_text = standings.format_win_rate_table()

    (results_dir / "win_rate_matrix.txt").write_text(table_text + "\n")
    with (results_dir / "win_rate_matrix.json").open("w") as f:
        json.dump(matrix, f, indent=2)
    with (results_dir / "arena_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(standings.summary_rows()[0].keys()))
        writer.writeheader()
        writer.writerows(standings.summary_rows())

    from connectx.submission.make_submission import export_alphazero_checkpoint, export_ppo_model, render_submission

    kind = args.kind
    if kind == "auto":
        if not ppo_ckpt or not ppo_ckpt.exists():
            kind = "alphazero"
        elif not az_ckpt or not az_ckpt.exists():
            kind = "ppo"
        else:
            kind = select_best_kind(az_ckpt, ppo_ckpt, games=args.arena_games, alphazero_simulations=args.alphazero_simulations)

    submission_dir = Path("submission")
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / "submission.py"
    if kind == "alphazero":
        weights_b64, _meta = export_alphazero_checkpoint(az_ckpt)
    else:
        if not ppo_ckpt or not ppo_ckpt.exists():
            raise RuntimeError("PPO checkpoint missing for submission export.")
        weights_b64, _meta = export_ppo_model(ppo_ckpt)

    submission_path.write_text(render_submission(kind, weights_b64))
    validate_submission(submission_path, games=2)

    (submission_dir / "submission_kind.txt").write_text(kind + "\n")
    print(f"Arena table written to {results_dir}")
    print(table_text)
    print(f"Submission kind: {kind}")
    print(f"Submission: {submission_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize ConnectX results after training.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--ppo-model", default=None)
    parser.add_argument("--alphazero-checkpoint", default=None)
    parser.add_argument("--dqn-model", default=None)
    parser.add_argument("--kind", choices=["auto", "alphazero", "ppo"], default="auto")
    parser.add_argument("--arena-games", type=int, default=100)
    parser.add_argument("--negamax-games", type=int, default=30)
    parser.add_argument("--alphazero-simulations", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    finalize(parse_args())
