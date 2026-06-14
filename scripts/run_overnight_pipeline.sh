#!/usr/bin/env bash
# Overnight pipeline: Prompt ① validation → Prompt ② hybrid build + validation
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="$ROOT/results/overnight_pipeline.log"
mkdir -p "$ROOT/results"

exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "ConnectX overnight pipeline started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Working directory: $ROOT"
echo "=============================================="

step() {
    echo ""
    echo ">>> [$1] $(date -u '+%H:%M:%S') $2"
}

fail() {
    echo "!!! PIPELINE FAILED at step: $1"
    echo "See $LOG"
    exit 1
}

# --- Prompt ① ---
step "P1" "Validate cached submission"
if ! python3 scripts/validate_cached.py; then
    fail "P1 validate_cached"
fi

if ! grep -q 'RESULT: PASS' "$ROOT/results/validate_cached.md" 2>/dev/null; then
    if ! grep -q '\*\*RESULT: PASS\*\*' "$ROOT/results/validate_cached.md" 2>/dev/null; then
        fail "P1 validate_cached.md not PASS"
    fi
fi
echo "Prompt ① PASS"

# --- Prompt ② diagnosis ---
step "P2a" "Solver diagnosis"
python3 scripts/diagnose_solver.py || fail "P2a diagnose_solver"

# --- Prompt ② build ---
step "P2b" "Build hybrid submission"
python3 -m connectx.submission.make_hybrid_submission || fail "P2b make_hybrid"

if [[ ! -f "$ROOT/submission/submission_alphazero_hybrid_gen8.py" ]]; then
    fail "P2b hybrid file missing"
fi
echo "Built submission/submission_alphazero_hybrid_gen8.py"

# --- Prompt ② validate (fast gates first) ---
step "P2c" "Hybrid validation (gates A + bitboard)"
if ! python3 scripts/validate_hybrid.py --skip-slow --skip-pmr; then
    echo "WARN: fast hybrid gates failed; continuing to full validation anyway"
fi

step "P2d" "Hybrid validation (full, includes PMR)"
if ! python3 scripts/validate_hybrid.py; then
    fail "P2d validate_hybrid"
fi

if ! grep -q '\*\*RESULT: PASS\*\*' "$ROOT/results/validate_hybrid.md" 2>/dev/null; then
    fail "P2d validate_hybrid.md not PASS"
fi

echo ""
echo "=============================================="
echo "PIPELINE COMPLETE: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  results/validate_cached.md"
echo "  results/solver_diagnosis.md"
echo "  results/validate_hybrid.md"
echo "  submission/submission_alphazero_hybrid_gen8.py"
echo "=============================================="
