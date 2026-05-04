#!/usr/bin/env bash
# =============================================================================
# scripts/04a_stub_sanity.sh
#
# Stub-backbone sanity check: 100 optimiser steps on synthetic data.
# Verifies:
#   1. The training loop completes without crash.
#   2. Loss is not NaN at any logged step.
#   3. Loss drops > 67 % from step-1 to step-100.
#
# All assertions are in tests/test_stub_sanity.py (run via pytest).
# This script additionally runs the Hydra training loop end-to-end to
# produce a real loss curve in results/stub_sanity/train_log.csv.
#
# Usage:
#   bash scripts/04a_stub_sanity.sh           # full run + pytest
#   bash scripts/04a_stub_sanity.sh --test-only   # pytest only (no train run)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "[ENV]  python=$(python --version 2>&1)  cwd=$(pwd)"
echo "[CFG]  100 steps, batch=4, synthetic_data=true, backbone=stub"
echo "================================================================"

# ---------------------------------------------------------------------------
# Phase 1: pytest unit tests (no data required — pure tensor arithmetic)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 1: pytest tests/test_stub_sanity.py ==="
python -m pytest tests/test_stub_sanity.py -v --tb=short
echo "[OK] Phase 1 passed"

if [[ "${1:-}" == "--test-only" ]]; then
    echo ""
    echo "[SKIP] --test-only flag set; skipping end-to-end training run."
    exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: End-to-end Hydra training run (synthetic, 100 steps)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 2: 100-step synthetic training run ==="
python -m otp.train.train_otp_soft \
    --config-name otp_soft_stub \
    output_dir=results/stub_sanity

echo ""
echo "=== Loss curve summary ==="
if [ -f results/stub_sanity/train_log.csv ]; then
    python - <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open("results/stub_sanity/train_log.csv")))
if not rows:
    print("[FAIL] train_log.csv is empty", file=sys.stderr)
    sys.exit(1)
losses = [float(r["total_loss"]) for r in rows]
first, last = losses[0], losses[-1]
drop = (1 - last / first) * 100 if first > 0 else 0
print(f"  steps logged : {len(losses)}")
print(f"  first loss   : {first:.4f}")
print(f"  last  loss   : {last:.4f}")
print(f"  drop         : {drop:.1f}%")
if any(float(l) != float(l) for l in losses):
    print("[FAIL] NaN detected in loss curve", file=sys.stderr)
    sys.exit(1)
if drop < 67:
    print(f"[WARN] loss drop {drop:.1f}% < 67% target (may need more steps or lr tuning)")
else:
    print(f"[OK]  loss drop {drop:.1f}% >= 67% target")
PYEOF
else
    echo "[WARN] train_log.csv not found — training may not have logged"
fi

echo ""
echo "================================================================"
echo "[OK] 04a_stub_sanity.sh complete"
echo "================================================================"
