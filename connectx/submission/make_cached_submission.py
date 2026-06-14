"""Add inference cache to rollback gen8 submission."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

CACHE_PREAMBLE = '''
_FORWARD_CACHE = {}
_FORWARD_CACHE_MAX = 50000


def _mirror_board_flat(board, rows, columns):
    mirrored = []
    for row in range(rows):
        for col in range(columns - 1, -1, -1):
            mirrored.append(int(board[row * columns + col]))
    return mirrored


def _forward(board, mark, rows, columns):
    flat = tuple(int(cell) for cell in board)
    mirrored = tuple(_mirror_board_flat(board, rows, columns))
    use_mirror = mirrored < flat
    key = (mirrored if use_mirror else flat, int(mark))
    cached = _FORWARD_CACHE.get(key)
    if cached is not None:
        logits, value = cached
        if use_mirror:
            logits = logits[::-1].copy()
        return logits, value
    logits, value = _forward_compute(board, mark, rows, columns)
    if len(_FORWARD_CACHE) < _FORWARD_CACHE_MAX:
        store_logits = logits[::-1].copy() if use_mirror else logits.copy()
        _FORWARD_CACHE[key] = (store_logits, float(value))
    return logits, value
'''

FORWARD_RENAME = "def _forward_compute(board, mark, rows, columns):"


def build_cached_submission(template_path: Path, output_path: Path) -> Path:
    text = template_path.read_text(encoding="utf-8")
    text = re.sub(
        r"# note:.*\n",
        "# note: gen8 rollback + forward cache (mirror-normalized, max 50000)\n",
        text,
        count=1,
    )
    text = text.replace("# tag: rollback_gen8", "# tag: gen8_cached")
    text = text.replace(
        "def _forward(board, mark, rows, columns):",
        FORWARD_RENAME,
        1,
    )
    if "_forward_compute" not in text:
        raise ValueError("failed to rename _forward in template")
    insert_at = text.index(FORWARD_RENAME)
    text = text[:insert_at] + CACHE_PREAMBLE.strip() + "\n\n\n" + text[insert_at:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("submission/submission_alphazero_rollback_gen8.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission/submission_alphazero_gen8_cached.py"),
    )
    args = parser.parse_args()
    path = build_cached_submission(args.template, args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
