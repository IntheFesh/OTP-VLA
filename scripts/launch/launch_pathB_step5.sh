#!/usr/bin/env bash
# =============================================================================
# Path B Step 5 launcher: sequential 5a + 5b after Stage 3 (OFT) completes.
#
# Usage:
#   nohup bash scripts/launch/launch_pathB_step5.sh \
#     > logs/launcher_step5_$(date +%Y%m%d_%H%M).log 2>&1 &
#
# Workflow:
#   1. Wait for Stage 3 (any "18_oft_unseen_phrasing.py" process) to exit
#   2. Sleep 60s for GPU memory cleanup
#   3. Launch Step 5a (base Path B, no Cocos)
#   4. Wait for 5a finish
#   5. Launch Step 5b (Cocos enabled)
#   6. Wait for 5b finish
#   7. Print summary: last 30 lines of each log + final losses
#
# Failure handling:
#   - If 5a crashes (non-zero exit), still try 5b (different architecture)
#   - All failures recorded; script never aborts early
# =============================================================================

set -u  # error on unset vars, but continue on cmd failures (NOT set -e)

REPO_ROOT="/root/autodl-tmp/OTP-VLA"
cd "$REPO_ROOT"

echo "=============================================="
echo "Path B Step 5 launcher started at $(date)"
echo "=============================================="

# === Phase 1: Wait for Stage 3 to complete ===================================
echo
echo "[1/4] Waiting for Stage 3 (OFT) to complete..."
WAIT_START=$(date +%s)
while true; do
    if ! ps aux | grep "18_oft_unseen_phrasing.py" | grep -v grep > /dev/null 2>&1; then
        echo "  Stage 3 not running — proceeding."
        break
    fi
    elapsed_min=$(( ($(date +%s) - WAIT_START) / 60 ))
    echo "  Stage 3 still running ($(date +%H:%M:%S), waited ${elapsed_min} min)"
    sleep 120  # check every 2 min
done

# Verify Stage 3 finished cleanly: result JSON should exist
if [[ -f results/phase0/oft_unseen_phrasing_full.json ]]; then
    echo "  Stage 3 result file present:"
    ls -la results/phase0/oft_unseen_phrasing_full.json
else
    echo "  WARNING: oft_unseen_phrasing_full.json NOT FOUND."
    echo "  Stage 3 may have crashed. Check partial file:"
    ls -la results/phase0/oft_unseen_phrasing_full_partial.json 2>/dev/null || echo "  (no partial file either)"
    echo "  Continuing to Step 5 anyway (GPU should now be free)."
fi

# Brief sleep for GPU memory cleanup
echo "  Sleeping 60s for GPU cleanup..."
sleep 60

# === Phase 2: Step 5a (base Path B, no Cocos) =================================
TS=$(date +%Y%m%d_%H%M%S)
LOG_5a="logs/pathB_step5a_${TS}.log"
echo
echo "[2/4] Launching Step 5a (no Cocos)..."
echo "  Log: $LOG_5a"
echo "  Started: $(date)"

python -u -m otp.train.train_otp_soft \
    --config-name=otp_soft_pathB_overfit \
    > "$LOG_5a" 2>&1
EXIT_5a=$?

echo "  Step 5a exited: $EXIT_5a"
echo "  Finished: $(date)"

# === Phase 3: Step 5b (Cocos enabled) =========================================
TS=$(date +%Y%m%d_%H%M%S)
LOG_5b="logs/pathB_step5b_${TS}.log"
echo
echo "[3/4] Launching Step 5b (Cocos)..."
echo "  Log: $LOG_5b"
echo "  Started: $(date)"

# Brief sleep for GPU memory cleanup between runs
sleep 30

python -u -m otp.train.train_otp_soft \
    --config-name=otp_soft_pathB_overfit_cocos \
    > "$LOG_5b" 2>&1
EXIT_5b=$?

echo "  Step 5b exited: $EXIT_5b"
echo "  Finished: $(date)"

# === Phase 4: Summary =========================================================
echo
echo "=============================================="
echo "Step 5 sequence complete at $(date)"
echo "=============================================="
echo
echo "Step 5a (no Cocos):  exit $EXIT_5a, log $LOG_5a"
echo "Step 5b (Cocos):     exit $EXIT_5b, log $LOG_5b"
echo
echo "--- Step 5a last 30 lines ---"
tail -30 "$LOG_5a"
echo
echo "--- Step 5b last 30 lines ---"
tail -30 "$LOG_5b"
echo
echo "--- Final loss comparison (last log_every entries) ---"
echo "Step 5a final losses:"
grep -E "(loss|step)" "$LOG_5a" | tail -10
echo
echo "Step 5b final losses:"
grep -E "(loss|step)" "$LOG_5b" | tail -10
echo
echo "Pass criterion (V7 §IV Step 5): L1 → 0.01 within 500 steps"
echo "Manual review required to determine which variant proceeds to Step 6."
