from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from connectx.agents.q_learning import TabularQAgent
from connectx.submission.make_submission import (
    _npz_b64,
    export_alphazero_checkpoint,
    export_ppo_model,
    render_submission,
    validate_submission,
)


def export_dqn_model(model_path: str | Path) -> tuple[str, dict]:
    import torch

    from connectx.agents.dqn import DQNAgent

    agent = DQNAgent.load(model_path, device="cpu")
    arrays = {
        key: value.detach().cpu().numpy()
        for key, value in agent.online.state_dict().items()
    }
    meta = {"kind": "dqn", "config": agent.config.__dict__}
    arrays["__meta__"] = np.asarray([json.dumps(meta)])
    return _npz_b64(arrays), meta


def export_q_learning_model(model_path: str | Path) -> tuple[str, dict]:
    agent = TabularQAgent.load(model_path)
    table = {str(key): value.astype(np.float32) for key, value in agent.q_table.items()}
    meta = {
        "kind": "q_learning",
        "config": {
            "rows": agent.config.rows,
            "columns": agent.config.columns,
            "inarow": agent.config.inarow,
        },
    }
    arrays = {
        "__meta__": np.asarray([json.dumps(meta)]),
        "q_table_pkl": np.frombuffer(pickle.dumps(table, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8),
    }
    return _npz_b64(arrays), meta


def best_alphazero_checkpoint() -> Path:
    from connectx.training.finalize_results import best_alphazero_from_runs

    path = best_alphazero_from_runs()
    if path is None:
        final = Path("runs/alphazero/checkpoints/alphazero_final.pt")
        if final.exists():
            return final
        overnight = Path("runs/alphazero_overnight/checkpoints/alphazero_final.pt")
        if overnight.exists():
            return overnight
        raise FileNotFoundError("No AlphaZero checkpoint found")
    return path


def best_ppo_checkpoint() -> Path:
    paths = sorted(Path("runs/ppo/checkpoints").glob("ppo_*.zip"))
    if not paths:
        raise FileNotFoundError("No PPO checkpoint found")
    return paths[-1]


DQN_TEMPLATE = r'''
import base64
import io
import json

import numpy as np

WEIGHTS_B64 = "__WEIGHTS_B64__"
PARAMS = None
META = None


def _load():
    global PARAMS, META
    if PARAMS is None:
        data = np.load(io.BytesIO(base64.b64decode(WEIGHTS_B64.encode("ascii"))), allow_pickle=False)
        PARAMS = {key: data[key] for key in data.files if key != "__meta__"}
        META = json.loads(str(data["__meta__"][0]))
    return PARAMS


def _obs_board_mark(obs):
    if isinstance(obs, dict):
        return [int(x) for x in obs["board"]], int(obs["mark"])
    return [int(x) for x in obs.board], int(obs.mark)


def _opp(mark):
    return 2 if mark == 1 else 1


def _mask(board, rows, columns):
    return np.asarray([board[col] == 0 for col in range(columns)], dtype=bool)


def _encode(board, mark, rows, columns):
    arr = np.asarray(board, dtype=np.int8).reshape(rows, columns)
    other = _opp(mark)
    return np.stack([(arr == mark).astype(np.float32), (arr == other).astype(np.float32)], axis=0)


def _conv2d(x, w, b, padding=1):
    if padding:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="constant")
    out_channels, in_channels, kh, kw = w.shape
    height = x.shape[1] - kh + 1
    width = x.shape[2] - kw + 1
    out = np.zeros((out_channels, height, width), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            patch = x[:, i:i + height, j:j + width]
            out += np.tensordot(w[:, :, i, j], patch, axes=([0], [0]))
    return out + b[:, None, None]


def _forward(board, mark, rows, columns):
    p = _load()
    x = _encode(board, mark, rows, columns)
    x = np.maximum(_conv2d(x, p["net.0.weight"], p["net.0.bias"], padding=1), 0)
    x = np.maximum(_conv2d(x, p["net.2.weight"], p["net.2.bias"], padding=1), 0)
    x = x.reshape(-1)
    x = np.maximum(p["net.4.weight"].dot(x) + p["net.4.bias"], 0)
    return p["net.6.weight"].dot(x) + p["net.6.bias"]


def agent(observation, configuration):
    rows = int(getattr(configuration, "rows", configuration.get("rows", 6) if isinstance(configuration, dict) else 6))
    columns = int(getattr(configuration, "columns", configuration.get("columns", 7) if isinstance(configuration, dict) else 7))
    board, mark = _obs_board_mark(observation)
    q = _forward(board, mark, rows, columns)
    mask = _mask(board, rows, columns)
    q = q.astype(np.float64)
    q[~mask] = -1e30
    return int(np.argmax(q)) if mask.any() else 0
'''


QLEARNING_TEMPLATE = r'''
import base64
import io
import json
import pickle

import numpy as np

WEIGHTS_B64 = "__WEIGHTS_B64__"
META = None
Q_TABLE = None


def _load():
    global META, Q_TABLE
    if Q_TABLE is None:
        data = np.load(io.BytesIO(base64.b64decode(WEIGHTS_B64.encode("ascii"))), allow_pickle=False)
        META = json.loads(str(data["__meta__"][0]))
        raw = pickle.loads(data["q_table_pkl"].tobytes())
        Q_TABLE = {eval(key): np.asarray(values, dtype=np.float32) for key, values in raw.items()}
    return META, Q_TABLE


def _obs_board_mark(obs):
    if isinstance(obs, dict):
        return [int(x) for x in obs["board"]], int(obs["mark"])
    return [int(x) for x in obs.board], int(obs.mark)


def _canonical(board, mark):
    opponent = 2 if mark == 1 else 1
    return tuple(1 if cell == mark else 2 if cell == opponent else 0 for cell in board)


def _mask(board, columns):
    return [col for col in range(columns) if board[col] == 0]


def agent(observation, configuration):
    meta, q_table = _load()
    cfg = meta["config"]
    rows, columns, inarow = int(cfg["rows"]), int(cfg["columns"]), int(cfg["inarow"])
    board, mark = _obs_board_mark(observation)
    if len(board) != rows * columns:
        return _mask(board, int(getattr(configuration, "columns", 7)))[0]
    state = _canonical(board, mark)
    legal = _mask(board, columns)
    if not legal:
        return 0
    values = q_table.get(state, np.zeros(columns, dtype=np.float32))
    best = max(legal, key=lambda col: float(values[col]))
    return int(best)
'''


def render_dqn(weights_b64: str) -> str:
    return DQN_TEMPLATE.replace("__WEIGHTS_B64__", weights_b64)


def render_q_learning(weights_b64: str) -> str:
    return QLEARNING_TEMPLATE.replace("__WEIGHTS_B64__", weights_b64)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_")


def submission_header(
    *,
    kind: str,
    source: Path,
    tag: str,
    notes: str,
    board: str = "6x7 connect-4",
) -> str:
    lines = [
        "# Kaggle ConnectX submission",
        f"# algorithm: {kind}",
        f"# board: {board}",
        f"# checkpoint: {source}",
        f"# tag: {tag}",
        f"# exported_at: {_utc_now()}",
    ]
    if notes:
        for note_line in notes.strip().splitlines():
            lines.append(f"# note: {note_line.strip()}")
    lines.append("")
    return "\n".join(lines)


def _board_for_kind(kind: str, meta: dict | None) -> str:
    if kind == "q_learning" and meta and "config" in meta:
        cfg = meta["config"]
        return f"{cfg['rows']}x{cfg['columns']} connect-{cfg['inarow']}"
    return "6x7 connect-4"


def _load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _append_registry(path: Path, entry: dict) -> None:
    registry = _load_registry(path)
    registry.append(entry)
    path.write_text(json.dumps(registry, indent=2) + "\n")


def _write_notes(output_dir: Path, registry: list[dict]) -> None:
    lines = [
        "ConnectX submission files (new exports go to versions/, existing files are kept by default).",
        "",
        "Upload to Kaggle: copy one agent() file; submission.py is only updated with --set-default.",
        "",
    ]
    for item in reversed(registry[-20:]):
        lines.append(
            f"- [{item['exported_at']}] {item['kind']} tag={item['tag']} "
            f"-> {item['path']} (source: {item['source']})"
        )
        if item.get("notes"):
            lines.append(f"  notes: {item['notes']}")
    (output_dir / "submission_notes.txt").write_text("\n".join(lines) + "\n")


def make_all_submissions(
    output_dir: str | Path,
    *,
    alphazero_checkpoint: Path | None = None,
    ppo_model: Path | None = None,
    dqn_model: Path | None = None,
    q_learning_model: Path | None = None,
    validate: bool = False,
    tag: str = "export",
    notes: str = "",
    overwrite_algo_files: bool = False,
    set_default: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    versions_dir = output_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    exported_at = _utc_now()
    tag_slug = _slug(tag)

    az_path = alphazero_checkpoint or best_alphazero_checkpoint()
    ppo_path = ppo_model or best_ppo_checkpoint()
    dqn_path = dqn_model or Path("runs/dqn/dqn.pt")
    ql_path = q_learning_model or Path("runs/q_learning/q_learning.pkl")

    sources = {
        "alphazero": az_path,
        "ppo": ppo_path,
        "dqn": dqn_path,
        "q_learning": ql_path,
    }

    payloads: list[tuple[str, str, dict | None]] = []

    az_b64, _ = export_alphazero_checkpoint(az_path)
    payloads.append(("alphazero", render_submission("alphazero", az_b64), None))

    ppo_b64, _ = export_ppo_model(ppo_path)
    payloads.append(("ppo", render_submission("ppo", ppo_b64), None))

    dqn_b64, _ = export_dqn_model(dqn_path)
    payloads.append(("dqn", render_dqn(dqn_b64), None))

    ql_b64, ql_meta = export_q_learning_model(ql_path)
    payloads.append(("q_learning", render_q_learning(ql_b64), ql_meta))

    written: dict[str, Path] = {}
    registry_path = output_dir / "submission_registry.json"
    manifest_entries: dict[str, dict] = {}

    for kind, body, meta in payloads:
        source = sources[kind]
        if not source.exists():
            continue
        board = _board_for_kind(kind, meta)
        header = submission_header(kind=kind, source=source, tag=tag, notes=notes, board=board)
        content = header + body.lstrip("\n")

        version_name = f"submission_{kind}_{tag_slug}_{exported_at.replace(':', '').replace('-', '')}.py"
        version_path = versions_dir / version_name
        version_path.write_text(content)
        written[kind] = version_path

        canonical = output_dir / f"submission_{kind}.py"
        if overwrite_algo_files or not canonical.exists():
            canonical.write_text(content)

        entry = {
            "kind": kind,
            "tag": tag,
            "notes": notes,
            "exported_at": exported_at,
            "path": str(version_path.relative_to(output_dir)),
            "canonical_path": str(canonical.relative_to(output_dir)) if canonical.exists() else "",
            "source": str(source),
            "board": board,
        }
        _append_registry(registry_path, entry)
        manifest_entries[kind] = entry

        if validate:
            if kind == "q_learning":
                cfg = ql_meta["config"] if ql_meta else {"rows": 4, "columns": 5, "inarow": 3}
                validate_submission(
                    version_path,
                    games=2,
                    configuration={**cfg, "actTimeout": 2, "timeout": 2},
                )
            else:
                validate_submission(version_path, games=2)

    if set_default and "alphazero" in written:
        default_path = output_dir / "submission.py"
        default_src = output_dir / "submission_alphazero.py"
        if default_src.exists():
            default_path.write_text(default_src.read_text())
        (output_dir / "submission_kind.txt").write_text(
            f"alphazero\n# tag: {tag}\n# source: {sources['alphazero']}\n# exported: {exported_at}\n"
        )

    manifest = {
        "latest_tag": tag,
        "latest_export": exported_at,
        "notes": notes,
        "overwrite_algo_files": overwrite_algo_files,
        "set_default": set_default,
        "algorithms": manifest_entries,
        "versions_dir": "versions",
    }
    (output_dir / "submissions_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_notes(output_dir, _load_registry(registry_path))
    return written


def wait_for_alphazero(timeout_sec: int = 7200, poll_sec: int = 60) -> None:
    final = Path("runs/alphazero_overnight/checkpoints/alphazero_final.pt")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if final.exists():
            return
        if not __import__("subprocess").run(["pgrep", "-f", "train_alphazero"], capture_output=True).returncode == 0:
            return
        time.sleep(poll_sec)
    raise TimeoutError(f"AlphaZero still running after {timeout_sec}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Kaggle submissions for all four RL algorithms.")
    parser.add_argument("--output-dir", default="submission")
    parser.add_argument("--alphazero-checkpoint", default=None)
    parser.add_argument("--ppo-model", default=None)
    parser.add_argument("--dqn-model", default=None)
    parser.add_argument("--q-learning-model", default=None)
    parser.add_argument("--wait-alphazero", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--tag", default="export", help="Version label for filenames and manifest (e.g. overnight_gen38, push).")
    parser.add_argument("--notes", default="", help="Free-form notes stored in headers and submission_notes.txt.")
    parser.add_argument(
        "--overwrite-algo-files",
        action="store_true",
        help="Replace submission_<algo>.py; default keeps existing and only adds versions/.",
    )
    parser.add_argument(
        "--set-default",
        action="store_true",
        help="Update submission.py from submission_alphazero.py (off by default).",
    )
    args = parser.parse_args()

    if args.wait_alphazero:
        print("Waiting for AlphaZero overnight to finish...", flush=True)
        wait_for_alphazero()

    paths = make_all_submissions(
        args.output_dir,
        alphazero_checkpoint=Path(args.alphazero_checkpoint) if args.alphazero_checkpoint else None,
        ppo_model=Path(args.ppo_model) if args.ppo_model else None,
        dqn_model=Path(args.dqn_model) if args.dqn_model else None,
        q_learning_model=Path(args.q_learning_model) if args.q_learning_model else None,
        validate=args.validate,
        tag=args.tag,
        notes=args.notes,
        overwrite_algo_files=args.overwrite_algo_files,
        set_default=args.set_default,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
