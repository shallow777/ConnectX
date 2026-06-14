"""生成 Kaggle 单文件 submission.py (single-file submission generator).

Kaggle 评测环境只有 NumPy 没有 torch/sb3, 且要求单文件提交, 所以这里的做法是:
1. 把训练好的权重保存成 npz 再 base64 编码, 直接嵌入到模板字符串里;
2. 模板内用纯 NumPy 手写前向传播 (conv/bn/linear 全部手动实现);
3. AlphaZero 模板还带一个按时间预算 (actTimeout) 跑的简化 MCTS;
4. 模板内同样实现了战术安全层 (先抢自己必胜点 / 再堵对手必胜点)。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from connectx.agents.alphazero.agent import make_alphazero_agent
from connectx.agents.ppo_selfplay import make_sb3_ppo_agent
from connectx.evaluation.arena import AgentSpec, evaluate_pair


def _npz_b64(arrays: dict[str, np.ndarray]) -> str:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_template_body(template_path: Path) -> list[str]:
    """抽取一个已生成 submission 模板里"非权重"的代码主体 (import + 推理逻辑),
    供 hybrid / solver 提交脚本复用; 跳过注释行和 WEIGHTS_B64 大块权重。"""
    lines = template_path.read_text(encoding="utf-8").splitlines()
    preamble: list[str] = []
    body: list[str] = []
    seen_weights = False
    for line in lines:
        if line.startswith("#"):
            continue
        if line.startswith("WEIGHTS_B64"):
            seen_weights = True
            continue
        if not seen_weights:
            if line.startswith("import ") or line == "":
                preamble.append(line)
            continue
        if line == "" and not body:
            continue
        body.append(line)
    combined = preamble + body
    while combined and combined[0] == "":
        combined.pop(0)
    if not combined:
        raise ValueError(f"Could not parse submission template: {template_path}")
    return combined


def export_alphazero_checkpoint(checkpoint_path: str | Path) -> tuple[str, dict[str, Any]]:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    # num_batches_tracked 是 BatchNorm 的训练统计计数, 推理用不到, 剔除省体积
    arrays = {
        key: value.detach().cpu().numpy()
        for key, value in payload["model_state_dict"].items()
        if not key.endswith("num_batches_tracked")
    }
    config = payload["network_config"]
    arrays["__meta__"] = np.asarray([json.dumps({"kind": "alphazero", "config": config})])
    return _npz_b64(arrays), config


def export_ppo_model(model_path: str | Path) -> tuple[str, dict[str, Any]]:
    try:
        from sb3_contrib import MaskablePPO
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("sb3-contrib and torch are required to export a PPO model on the server.") from exc

    model = MaskablePPO.load(str(model_path), device="cpu")
    arrays: dict[str, np.ndarray] = {}
    activations: list[str] = []
    linear_idx = 0
    for module in model.policy.mlp_extractor.policy_net:
        if isinstance(module, nn.Linear):
            arrays[f"policy.{linear_idx}.weight"] = module.weight.detach().cpu().numpy()
            arrays[f"policy.{linear_idx}.bias"] = module.bias.detach().cpu().numpy()
            linear_idx += 1
        elif isinstance(module, nn.Tanh):
            activations.append("tanh")
        elif isinstance(module, nn.ReLU):
            activations.append("relu")
        else:
            raise TypeError(f"Unsupported PPO policy module for numpy export: {module!r}")

    arrays["action.weight"] = model.policy.action_net.weight.detach().cpu().numpy()
    arrays["action.bias"] = model.policy.action_net.bias.detach().cpu().numpy()
    feature_keys = list(getattr(model.policy.features_extractor, "extractors", {}).keys())
    if not feature_keys:
        feature_keys = ["observation", "action_mask"]
    meta = {
        "kind": "ppo",
        "feature_keys": feature_keys,
        "activations": activations,
        "linear_layers": linear_idx,
    }
    arrays["__meta__"] = np.asarray([json.dumps(meta)])
    return _npz_b64(arrays), meta


def select_best_kind(
    alphazero_checkpoint: str | Path,
    ppo_model: str | Path,
    *,
    games: int = 200,
    alphazero_simulations: int = 80,
) -> str:
    """本地打 arena 决定提交哪个 agent (--kind auto 时使用)."""
    alphazero = AgentSpec(
        "alphazero",
        make_alphazero_agent(alphazero_checkpoint, simulations=alphazero_simulations, device="cpu", tactical_safety=True),
    )
    ppo = AgentSpec("ppo", make_sb3_ppo_agent(ppo_model, deterministic=True))
    stats = evaluate_pair(alphazero, ppo, games=games)
    return "alphazero" if stats.win_rate("alphazero") >= stats.win_rate("ppo") else "ppo"


ALPHAZERO_TEMPLATE = r'''
import base64
import io
import json
import time

import numpy as np


WEIGHTS_B64 = "__WEIGHTS_B64__"
PARAMS = None
META = None
PUCT_C = 2.0


def _cfg(config):
    if isinstance(config, dict):
        timeout = config.get("actTimeout", config.get("timeout", 2.0))
        return {
            "rows": int(config.get("rows", 6)),
            "columns": int(config.get("columns", 7)),
            "inarow": int(config.get("inarow", 4)),
            "timeout": float(timeout),
        }
    timeout = getattr(config, "actTimeout", getattr(config, "timeout", 2.0))
    return {
        "rows": int(getattr(config, "rows", 6)),
        "columns": int(getattr(config, "columns", 7)),
        "inarow": int(getattr(config, "inarow", 4)),
        "timeout": float(timeout),
    }


def _load():
    global PARAMS, META
    if PARAMS is None:
        raw = base64.b64decode(WEIGHTS_B64.encode("ascii"))
        data = np.load(io.BytesIO(raw), allow_pickle=False)
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


def _drop(board, action, mark, rows, columns):
    new = list(board)
    for row in range(rows - 1, -1, -1):
        idx = row * columns + action
        if new[idx] == 0:
            new[idx] = mark
            return new
    return new


def _winner(board, mark, rows, columns, inarow):
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(rows):
        for col in range(columns):
            if board[row * columns + col] != mark:
                continue
            for dr, dc in directions:
                end_row = row + (inarow - 1) * dr
                end_col = col + (inarow - 1) * dc
                if end_row < 0 or end_row >= rows or end_col < 0 or end_col >= columns:
                    continue
                ok = True
                for offset in range(inarow):
                    if board[(row + offset * dr) * columns + col + offset * dc] != mark:
                        ok = False
                        break
                if ok:
                    return True
    return False


def _tactical(board, mark, rows, columns, inarow):
    legal = [a for a, valid in enumerate(_mask(board, rows, columns)) if valid]
    for action in legal:
        if _winner(_drop(board, action, mark, rows, columns), mark, rows, columns, inarow):
            return action
    other = _opp(mark)
    for action in legal:
        if _winner(_drop(board, action, other, rows, columns), other, rows, columns, inarow):
            return action
    return None


def _encode(board, mark, rows, columns):
    arr = np.asarray(board, dtype=np.int8).reshape(rows, columns)
    other = _opp(mark)
    return np.stack([(arr == mark).astype(np.float32), (arr == other).astype(np.float32)], axis=0)


def _conv2d(x, w, b=None, padding=0):
    if padding:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode="constant")
    out_channels, in_channels, kh, kw = w.shape
    height = x.shape[1] - kh + 1
    width = x.shape[2] - kw + 1
    out = np.zeros((out_channels, height, width), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            patch = x[:, i:i + height, j:j + width]
            out += np.tensordot(w[:, :, i, j], patch, axes=([1], [0]))
    if b is not None:
        out += b[:, None, None]
    return out


def _bn(x, prefix, p):
    weight = p[prefix + ".weight"][:, None, None]
    bias = p[prefix + ".bias"][:, None, None]
    mean = p[prefix + ".running_mean"][:, None, None]
    var = p[prefix + ".running_var"][:, None, None]
    return (x - mean) / np.sqrt(var + 1e-5) * weight + bias


def _linear(x, weight, bias):
    return weight.dot(x) + bias


def _softmax_masked(logits, mask):
    probs = np.zeros_like(logits, dtype=np.float64)
    if not mask.any():
        return probs
    valid = logits[mask].astype(np.float64)
    valid = valid - np.max(valid)
    exp = np.exp(valid)
    probs[mask] = exp / exp.sum()
    return probs.astype(np.float32)


def _forward(board, mark, rows, columns):
    p = _load()
    x = _encode(board, mark, rows, columns)
    x = np.maximum(_bn(_conv2d(x, p["stem.0.weight"], padding=1), "stem.1", p), 0)
    blocks = int(META["config"]["residual_blocks"])
    for block_idx in range(blocks):
        residual = x
        prefix = "blocks.%d" % block_idx
        y = np.maximum(_bn(_conv2d(x, p[prefix + ".conv1.weight"], padding=1), prefix + ".bn1", p), 0)
        y = _bn(_conv2d(y, p[prefix + ".conv2.weight"], padding=1), prefix + ".bn2", p)
        x = np.maximum(y + residual, 0)

    pi = np.maximum(_bn(_conv2d(x, p["policy_head.0.weight"]), "policy_head.1", p), 0).reshape(-1)
    logits = _linear(pi, p["policy_head.4.weight"], p["policy_head.4.bias"])

    v = np.maximum(_bn(_conv2d(x, p["value_head.0.weight"]), "value_head.1", p), 0).reshape(-1)
    v = np.maximum(_linear(v, p["value_head.4.weight"], p["value_head.4.bias"]), 0)
    value = np.tanh(_linear(v, p["value_head.6.weight"], p["value_head.6.bias"]))[0]
    return logits.astype(np.float32), float(value)


class _Node:
    __slots__ = ("prior", "visit", "value_sum", "children")

    def __init__(self, prior):
        self.prior = float(prior)
        self.visit = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def value(self):
        return 0.0 if self.visit == 0 else self.value_sum / self.visit


def _expand(node, board, mark, rows, columns):
    logits, value = _forward(board, mark, rows, columns)
    mask = _mask(board, rows, columns)
    priors = _softmax_masked(logits, mask)
    if priors.sum() <= 0 and mask.any():
        priors[mask] = 1.0 / mask.sum()
    for action, valid in enumerate(mask):
        if valid:
            node.children[action] = _Node(priors[action])
    return value


def _select(node):
    parent_visits = max(1, node.visit)
    best_action = None
    best_child = None
    best_score = -1e30
    for action, child in node.children.items():
        score = -child.value + PUCT_C * np.sqrt(parent_visits) * child.prior / (1 + child.visit)
        if score > best_score:
            best_action = action
            best_child = child
            best_score = score
    return best_action, best_child


def _search(board, mark, rows, columns, inarow, deadline):
    root = _Node(1.0)
    _expand(root, board, mark, rows, columns)
    while time.time() < deadline and root.children:
        node = root
        scratch = list(board)
        to_play = mark
        path = [node]
        while node.children:
            action, node = _select(node)
            scratch = _drop(scratch, action, to_play, rows, columns)
            to_play = _opp(to_play)
            path.append(node)
        previous = _opp(to_play)
        if _winner(scratch, previous, rows, columns, inarow):
            value = -1.0
        elif not _mask(scratch, rows, columns).any():
            value = 0.0
        else:
            value = _expand(node, scratch, to_play, rows, columns)
        for item in reversed(path):
            item.visit += 1
            item.value_sum += value
            value = -value
    if not root.children:
        legal = np.flatnonzero(_mask(board, rows, columns))
        return int(legal[0]) if legal.size else 0
    return max(root.children.items(), key=lambda kv: kv[1].visit)[0]


def agent(observation, configuration):
    cfg = _cfg(configuration)
    rows, columns, inarow = cfg["rows"], cfg["columns"], cfg["inarow"]
    board, mark = _obs_board_mark(observation)
    action = _tactical(board, mark, rows, columns, inarow)
    if action is not None:
        return int(action)
    budget = max(0.05, min(0.80, cfg["timeout"] * 0.45))
    deadline = time.time() + budget
    action = _search(board, mark, rows, columns, inarow, deadline)
    if not _mask(board, rows, columns)[action]:
        legal = np.flatnonzero(_mask(board, rows, columns))
        return int(legal[0]) if legal.size else 0
    return int(action)
'''


PPO_TEMPLATE = r'''
import base64
import io
import json

import numpy as np


WEIGHTS_B64 = "__WEIGHTS_B64__"
PARAMS = None
META = None


def _cfg(config):
    if isinstance(config, dict):
        return {
            "rows": int(config.get("rows", 6)),
            "columns": int(config.get("columns", 7)),
            "inarow": int(config.get("inarow", 4)),
        }
    return {
        "rows": int(getattr(config, "rows", 6)),
        "columns": int(getattr(config, "columns", 7)),
        "inarow": int(getattr(config, "inarow", 4)),
    }


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


def _drop(board, action, mark, rows, columns):
    new = list(board)
    for row in range(rows - 1, -1, -1):
        idx = row * columns + action
        if new[idx] == 0:
            new[idx] = mark
            return new
    return new


def _winner(board, mark, rows, columns, inarow):
    for row in range(rows):
        for col in range(columns):
            if board[row * columns + col] != mark:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_row = row + (inarow - 1) * dr
                end_col = col + (inarow - 1) * dc
                if end_row < 0 or end_row >= rows or end_col < 0 or end_col >= columns:
                    continue
                if all(board[(row + k * dr) * columns + col + k * dc] == mark for k in range(inarow)):
                    return True
    return False


def _tactical(board, mark, rows, columns, inarow):
    legal = [a for a, valid in enumerate(_mask(board, rows, columns)) if valid]
    for action in legal:
        if _winner(_drop(board, action, mark, rows, columns), mark, rows, columns, inarow):
            return action
    other = _opp(mark)
    for action in legal:
        if _winner(_drop(board, action, other, rows, columns), other, rows, columns, inarow):
            return action
    return None


def _encode(board, mark, rows, columns):
    arr = np.asarray(board, dtype=np.int8).reshape(rows, columns)
    other = _opp(mark)
    return np.stack([(arr == mark).astype(np.float32), (arr == other).astype(np.float32)], axis=0)


def _features(board, mark, rows, columns):
    _load()
    mask = _mask(board, rows, columns).astype(np.float32)
    encoded = _encode(board, mark, rows, columns)
    parts = []
    for key in META["feature_keys"]:
        if key == "observation":
            parts.append(encoded.reshape(-1))
        elif key == "action_mask":
            parts.append(mask.reshape(-1))
    return np.concatenate(parts).astype(np.float32), mask.astype(bool)


def _forward(features):
    p = _load()
    x = features
    for idx in range(int(META["linear_layers"])):
        x = p["policy.%d.weight" % idx].dot(x) + p["policy.%d.bias" % idx]
        if idx < len(META["activations"]):
            act = META["activations"][idx]
            if act == "tanh":
                x = np.tanh(x)
            elif act == "relu":
                x = np.maximum(x, 0)
    return p["action.weight"].dot(x) + p["action.bias"]


def agent(observation, configuration):
    cfg = _cfg(configuration)
    rows, columns, inarow = cfg["rows"], cfg["columns"], cfg["inarow"]
    board, mark = _obs_board_mark(observation)
    action = _tactical(board, mark, rows, columns, inarow)
    if action is not None:
        return int(action)
    features, mask = _features(board, mark, rows, columns)
    logits = _forward(features)
    logits = logits.astype(np.float64)
    logits[~mask] = -1e30
    if not mask.any():
        return 0
    return int(np.argmax(logits))
'''


def render_submission(kind: str, weights_b64: str) -> str:
    if kind == "alphazero":
        return ALPHAZERO_TEMPLATE.replace("__WEIGHTS_B64__", weights_b64)
    if kind == "ppo":
        return PPO_TEMPLATE.replace("__WEIGHTS_B64__", weights_b64)
    raise ValueError(f"Unknown submission kind: {kind}")


def validate_submission(path: str | Path, games: int = 2, configuration: dict | None = None) -> None:
    try:
        from kaggle_environments import evaluate
    except ImportError as exc:
        raise RuntimeError("kaggle_environments is required for local submission validation") from exc

    namespace: dict[str, Any] = {}
    exec(Path(path).read_text(), namespace)
    config = configuration or {"rows": 6, "columns": 7, "inarow": 4, "actTimeout": 2, "timeout": 2}
    evaluate(
        "connectx",
        [namespace["agent"], "negamax"],
        configuration=config,
        num_episodes=games,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kaggle ConnectX single-file submission.py.")
    parser.add_argument("--kind", choices=["alphazero", "ppo", "auto"], required=True)
    parser.add_argument("--alphazero-checkpoint", default=None)
    parser.add_argument("--ppo-model", default=None)
    parser.add_argument("--output", default="submission.py")
    parser.add_argument("--arena-games", type=int, default=200)
    parser.add_argument("--alphazero-simulations", type=int, default=80)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    kind = args.kind
    if kind == "auto":
        if not args.alphazero_checkpoint or not args.ppo_model:
            raise ValueError("--kind auto requires both --alphazero-checkpoint and --ppo-model")
        kind = select_best_kind(
            args.alphazero_checkpoint,
            args.ppo_model,
            games=args.arena_games,
            alphazero_simulations=args.alphazero_simulations,
        )

    if kind == "alphazero":
        if not args.alphazero_checkpoint:
            raise ValueError("--kind alphazero requires --alphazero-checkpoint")
        weights_b64, _meta = export_alphazero_checkpoint(args.alphazero_checkpoint)
    elif kind == "ppo":
        if not args.ppo_model:
            raise ValueError("--kind ppo requires --ppo-model")
        weights_b64, _meta = export_ppo_model(args.ppo_model)
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    output = Path(args.output)
    output.write_text(render_submission(kind, weights_b64))
    if args.validate:
        validate_submission(output)


if __name__ == "__main__":
    main()
