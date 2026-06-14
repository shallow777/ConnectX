# Overnight Pipeline Status

**Started:** 2026-06-12 (tmux session `connectx`)

## Attach to watch progress

```bash
tmux attach -t connectx
# detach: Ctrl+B then D
```

## Log file

```bash
tail -f /root/autodl-tmp/ConnectX_new/results/overnight_pipeline.log
```

## Pipeline steps

1. **P1** `scripts/validate_cached.py` → `results/validate_cached.md`
2. **P2a** `scripts/diagnose_solver.py` → `results/solver_diagnosis.md`
3. **P2b** `python3 -m connectx.submission.make_hybrid_submission` → `submission/submission_alphazero_hybrid_gen8.py`
4. **P2c/P2d** `scripts/validate_hybrid.py` → `results/validate_hybrid.md`

## Outputs when complete

- `submission/submission_alphazero_gen8_cached.py` (Prompt ①)
- `submission/submission_alphazero_hybrid_gen8.py` (Prompt ②)
- `results/validate_cached.md`
- `results/solver_diagnosis.md`
- `results/validate_hybrid.md`
