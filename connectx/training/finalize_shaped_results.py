from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
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
from connectx.agents.reward_shaping import RewardShapingConfig
from connectx.evaluation.arena import AgentSpec, evaluate_agents
from connectx.evaluation.kaggle_eval import evaluate_against_negamax
from connectx.training.finalize_results import (
    copy_curve,
    latest_checkpoint,
    negamax_row,
    plot_csv,
    plot_negamax_comparison,
)
from connectx.training.run_manifest import append_training_journal


def best_shaped_ppo() -> Path | None:
    return latest_checkpoint(Path("runs/ppo_shaped/checkpoints"), "ppo_*.zip")


def best_shaped_alphazero() -> Path | None:
    ckpt_dir = Path("runs/alphazero_shaped/checkpoints")
    final = ckpt_dir / "alphazero_final.pt"
    if final.exists():
        return final
    curve = Path("runs/alphazero_shaped/negamax_curve.csv")
    if not curve.exists():
        return None
    frame = pd.read_csv(curve)
    if frame.empty or "negamax_win_rate" not in frame.columns:
        return None
    row = frame.loc[frame["negamax_win_rate"].idxmax()]
    generation = int(row["generation"])
    accepted = ckpt_dir / f"generation_{generation:04d}_accepted.pt"
    if accepted.exists():
        return accepted
    champion = ckpt_dir / f"generation_{generation:04d}_champion.pt"
    return champion if champion.exists() else None


def load_baseline_comparison(results_dir: Path) -> pd.DataFrame | None:
    baseline_path = Path("results/negamax_final_comparison.csv")
    if not baseline_path.exists():
        return None
    baseline = pd.read_csv(baseline_path)
    baseline["variant"] = "baseline"
    return baseline


def plot_shaped_vs_baseline(shaped_rows: list[dict[str, Any]], output: Path) -> None:
    shaped = pd.DataFrame(shaped_rows)
    shaped["variant"] = "reward_shaping"
    baseline = load_baseline_comparison(Path("results"))
    if baseline is None:
        plot_negamax_comparison(shaped_rows, output)
        return
    keep = {"ppo", "dqn", "q_learning", "alphazero_main", "alphazero_best"}
    baseline = baseline[baseline["algorithm"].isin(keep)].copy()
    baseline = baseline.rename(columns={"algorithm": "algorithm_key"})
    mapping = {
        "ppo": "ppo",
        "dqn": "dqn",
        "q_learning": "q_learning",
        "alphazero_shaped": "alphazero_main",
    }
    shaped["algorithm_key"] = shaped["algorithm"].map(lambda name: mapping.get(name, name))
    merged = pd.concat(
        [
            baseline[["algorithm_key", "variant", "negamax_win_rate", "board"]],
            shaped[["algorithm_key", "variant", "negamax_win_rate", "board"]],
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(10, 4))
    sns.barplot(data=merged, x="algorithm_key", y="negamax_win_rate", hue="variant")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Negamax win rate")
    plt.title("Reward shaping vs baseline")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_shaped_training_curves(results_dir: Path) -> None:
    curves = results_dir / "curves"
    plot_csv(curves / "q_learning_shaped_learning_curve.csv", results_dir / "q_learning_shaped_curve.png", x="episode", y="q_states", title="Q-learning (shaped) table size")
    plot_csv(curves / "dqn_shaped_learning_curve.csv", results_dir / "dqn_shaped_learning_curve.png", x="episode", y="mean_loss", title="DQN (shaped) mean loss")
    plot_csv(curves / "ppo_shaped_negamax_curve.csv", results_dir / "ppo_shaped_negamax_curve.png", x="timesteps", y="negamax_win_rate", title="PPO (shaped) vs Negamax")
    plot_csv(
        curves / "alphazero_shaped_negamax_curve.csv",
        results_dir / "alphazero_shaped_negamax_curve.png",
        x="generation",
        y="negamax_win_rate",
        title="AlphaZero (shaped) vs Negamax",
    )
    dqn_path = curves / "dqn_shaped_learning_curve.csv"
    if dqn_path.exists():
        frame = pd.read_csv(dqn_path)
        if "winner" in frame.columns:
            frame["learner_win"] = (frame["winner"] == 1.0).astype(float)
            frame["rolling_win_rate"] = frame["learner_win"].rolling(500, min_periods=50).mean()
            plt.figure(figsize=(8, 4))
            sns.lineplot(data=frame, x="episode", y="rolling_win_rate")
            plt.title("DQN (shaped) self-play rolling win rate")
            plt.tight_layout()
            plt.savefig(results_dir / "dqn_shaped_selfplay_win_rate.png", dpi=150)
            plt.close()


def copy_training_logs(results_dir: Path) -> None:
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "train_shaped.log",
        "train_alphazero_shaped.log",
        "q_learning_shaped.log",
        "dqn_shaped.log",
        "ppo_shaped.log",
        "alphazero_shaped.log",
        "export_shaped.log",
    ):
        src = Path("logs") / pattern
        if src.exists() and src.stat().st_size > 0:
            shutil.copy2(src, logs_dir / pattern)


def write_training_report(results_dir: Path, negamax_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Reward Shaping Training Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Reward shaping config",
        "",
        "```json",
        json.dumps({"reward_shaping_config": __import__("dataclasses").asdict(RewardShapingConfig())}, indent=2),
        "```",
        "",
        "## Final negamax comparison",
        "",
    ]
    if negamax_rows:
        lines.append("```")
        lines.append(pd.DataFrame(negamax_rows).to_string(index=False))
        lines.append("```")
    else:
        lines.append("_No negamax results yet._")
    lines.extend(["", "## Training metrics", ""])
    if summary_rows:
        lines.append("```")
        lines.append(pd.DataFrame(summary_rows).to_string(index=False))
        lines.append("```")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Curves: `{results_dir / 'curves'}`",
            f"- Submissions: `submission/`",
            f"- Journal: `{results_dir / 'training_journal.csv'}`",
            f"- Logs: `{results_dir / 'logs'}`",
        ]
    )
    (results_dir / "training_report.md").write_text("\n".join(lines) + "\n")


def finalize_shaped(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    curves_dir = results_dir / "curves"
    results_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    ql_ckpt = Path(args.q_learning_model)
    dqn_ckpt = Path(args.dqn_model)
    ppo_ckpt = Path(args.ppo_model) if args.ppo_model else best_shaped_ppo()
    az_ckpt = Path(args.alphazero_checkpoint) if args.alphazero_checkpoint else best_shaped_alphazero()

    copy_curve(Path("runs/q_learning_shaped/learning_curve.csv"), curves_dir / "q_learning_shaped_learning_curve.csv")
    copy_curve(Path("runs/dqn_shaped/learning_curve.csv"), curves_dir / "dqn_shaped_learning_curve.csv")
    copy_curve(Path("runs/ppo_shaped/negamax_curve.csv"), curves_dir / "ppo_shaped_negamax_curve.csv")
    copy_curve(Path("runs/alphazero_shaped/negamax_curve.csv"), curves_dir / "alphazero_shaped_negamax_curve.csv")
    copy_training_logs(results_dir)
    plot_shaped_training_curves(results_dir)

    negamax_rows: list[dict[str, Any]] = []
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
    if dqn_ckpt.exists():
        negamax_rows.append(
            negamax_row(
                "dqn",
                evaluate_against_negamax(make_dqn_agent(dqn_ckpt, device="cpu"), games=args.negamax_games),
                board="6x7 connect-4",
                checkpoint=str(dqn_ckpt),
            )
        )
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
    if az_ckpt and az_ckpt.exists():
        negamax_rows.append(
            negamax_row(
                "alphazero_shaped",
                evaluate_against_negamax(
                    make_alphazero_agent(az_ckpt, simulations=args.alphazero_simulations, device="cpu", tactical_safety=True),
                    games=args.negamax_games,
                ),
                board="6x7 connect-4",
                checkpoint=str(az_ckpt),
            )
        )

    for row in negamax_rows:
        row["variant"] = "reward_shaping"

    with (results_dir / "negamax_comparison.csv").open("w", newline="") as handle:
        if negamax_rows:
            writer = csv.DictWriter(handle, fieldnames=list(negamax_rows[0].keys()))
            writer.writeheader()
            writer.writerows(negamax_rows)
    plot_negamax_comparison(negamax_rows, results_dir / "negamax_comparison.png")
    plot_shaped_vs_baseline(negamax_rows, results_dir / "shaped_vs_baseline.png")

    summary_rows: list[dict[str, Any]] = []
    for row in negamax_rows:
        summary_rows.append(
            {
                "algorithm": row["algorithm"],
                "variant": "reward_shaping",
                "board": row["board"],
                "checkpoint": row["checkpoint"],
                "metric": "negamax_win_rate",
                "metric_value": row["negamax_win_rate"],
            }
        )
    for curve_name, algorithm, metric_col, metric_name in (
        ("ppo_shaped_negamax_curve.csv", "ppo", "negamax_win_rate", "peak_negamax_win_rate"),
        ("alphazero_shaped_negamax_curve.csv", "alphazero_shaped", "negamax_win_rate", "peak_negamax_win_rate"),
        ("q_learning_shaped_learning_curve.csv", "q_learning", "q_states", "final_q_table_states"),
        ("dqn_shaped_learning_curve.csv", "dqn", "mean_loss", "final_mean_loss"),
    ):
        curve_path = curves_dir / curve_name
        if not curve_path.exists():
            continue
        frame = pd.read_csv(curve_path)
        if frame.empty or metric_col not in frame.columns:
            continue
        summary_rows.append(
            {
                "algorithm": algorithm,
                "variant": "reward_shaping",
                "board": "",
                "checkpoint": "",
                "metric": metric_name,
                "metric_value": float(frame[metric_col].iloc[-1] if metric_name.startswith("final") else frame[metric_col].max()),
            }
        )

    with (results_dir / "algorithms_summary.csv").open("w", newline="") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    (results_dir / "algorithms_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")

    manifest = {
        "variant": "reward_shaping",
        "reward_shaping_config": __import__("dataclasses").asdict(RewardShapingConfig()),
        "q_learning": {"checkpoint": str(ql_ckpt), "curve": str(curves_dir / "q_learning_shaped_learning_curve.csv")},
        "dqn": {"checkpoint": str(dqn_ckpt), "curve": str(curves_dir / "dqn_shaped_learning_curve.csv")},
        "ppo": {"checkpoint": str(ppo_ckpt) if ppo_ckpt else "", "curve": str(curves_dir / "ppo_shaped_negamax_curve.csv")},
        "alphazero_shaped": {
            "checkpoint": str(az_ckpt) if az_ckpt else "",
            "curve": str(curves_dir / "alphazero_shaped_negamax_curve.csv"),
        },
        "submissions_dir": "submission",
        "training_journal": str(results_dir / "training_journal.csv"),
    }
    (results_dir / "algorithms_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    agents: list[AgentSpec] = []
    if ppo_ckpt and ppo_ckpt.exists():
        agents.append(AgentSpec("ppo_shaped", wrap_with_tactical_safety(make_sb3_ppo_agent(ppo_ckpt))))
    if az_ckpt and az_ckpt.exists():
        agents.append(
            AgentSpec(
                "alphazero_shaped",
                make_alphazero_agent(az_ckpt, simulations=args.alphazero_simulations, device="cpu", tactical_safety=True),
            )
        )
    if dqn_ckpt.exists():
        agents.append(AgentSpec("dqn_shaped", make_dqn_agent(dqn_ckpt, device="cpu")))

    if len(agents) >= 2:
        standings = evaluate_agents(agents, games_per_pair=args.arena_games)
        matrix = standings.win_rate_matrix()
        (results_dir / "win_rate_matrix.txt").write_text(standings.format_win_rate_table() + "\n")
        with (results_dir / "win_rate_matrix.json").open("w") as handle:
            json.dump(matrix, handle, indent=2)
        with (results_dir / "arena_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(standings.summary_rows()[0].keys()))
            writer.writeheader()
            writer.writerows(standings.summary_rows())

    write_training_report(results_dir, negamax_rows, summary_rows)
    append_training_journal(
        results_dir / "training_journal.csv",
        stage="finalize_shaped",
        status="completed",
        run_dir=results_dir,
        details={"negamax_algorithms": len(negamax_rows), "arena_agents": len(agents)},
    )
    print(f"Shaped results written to {results_dir}")
    if negamax_rows:
        print(pd.DataFrame(negamax_rows)[["algorithm", "negamax_win_rate", "checkpoint"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize reward-shaping training results.")
    parser.add_argument("--results-dir", default="results/shaped")
    parser.add_argument("--q-learning-model", default="runs/q_learning_shaped/q_learning.pkl")
    parser.add_argument("--dqn-model", default="runs/dqn_shaped/dqn.pt")
    parser.add_argument("--ppo-model", default=None)
    parser.add_argument("--alphazero-checkpoint", default=None)
    parser.add_argument("--arena-games", type=int, default=100)
    parser.add_argument("--negamax-games", type=int, default=30)
    parser.add_argument("--alphazero-simulations", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    finalize_shaped(parse_args())
