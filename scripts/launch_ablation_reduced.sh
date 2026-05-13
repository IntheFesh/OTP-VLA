#!/bin/bash
# scripts/launch_ablation_reduced.sh
#
# Sequential launch of §V.D ablation retrains — REDUCED CELL LIST per Option α.
#
# Background: 21-cell launch (launch_ablation.sh full) was killed at 17:13 after
# budget re-audit. Cell 1 (pathB_seed42, PID 5666) survived parent kill via nohup
# detachment and continues independently. This reduced script launches the
# remaining 12 cells sequentially.
#
# Reduced plan (per launch_decision_memo.md Option α + post-audit revisions):
#   - Path B × 3 seeds (main claim, stat backing): cell 1 already running
#   - V3 baseline × 3 seeds (anchor for Path B comparison)
#   - Drop mesh × 3 seeds (variance anchor for drop ablations)
#   - Drop grasp × 1 seed (directional probe)
#   - Drop proprio × 1 seed (directional probe)
#   - Path B + Cocos × 0 (removed from paper, dropped)
#   - LIBERO-Goal M3 (cross-bench probe, ~1 hr)
#   - Variant (c) seed 42 × 1 (decoder family axis, mini E2E already verified Day 5)
#
# Usage:
#   bash scripts/launch_ablation_reduced.sh
#
# Features:
#   - Sequential GPU usage (one cell at a time)
#   - Resume capability (skip cells with ckpt at near-final step)
#   - Disk safety gate (abort if free < 50 GB before launching new cell)
#   - PID tracking and per-cell logging
#
# Pre-registration: notes/v_d_ablation_preregistration_2026-05-13.md
# Decision memo: notes/launch_decision_memo.md

set -uo pipefail

REPO_ROOT="/root/autodl-tmp/OTP-VLA"
cd "$REPO_ROOT" || { echo "ERR: cannot cd to $REPO_ROOT"; exit 1; }

MIN_FREE_DISK_GB=50

# ===========================================================================
# Cell list (12 cells; cell 1 pathB_seed42 already running independently)
# ===========================================================================

CONFIGS=(
    # Path B remaining seeds
    "otp_soft_pathB_seed137"
    "otp_soft_pathB_seed2026"
    # V3 baseline × 3 seeds
    "otp_soft_ablation_v3_seed42"
    "otp_soft_ablation_v3_seed137"
    "otp_soft_ablation_v3_seed2026"
    # Drop mesh × 3 seeds (variance anchor)
    "otp_soft_ablation_drop_mesh_seed42"
    "otp_soft_ablation_drop_mesh_seed137"
    "otp_soft_ablation_drop_mesh_seed2026"
    # Drop grasp × 1 (directional probe)
    "otp_soft_ablation_drop_grasp_seed42"
    # Drop proprio × 1 (directional probe)
    "otp_soft_ablation_drop_proprio_seed42"
    # Variant (c) seed 42 — mini E2E verified Day 5
    "otp_soft_ablation_variantc_seed42"
)

# Note: LIBERO-Goal M3 is a separate script, not a config-based training cell.
# It will be run manually as final step (see post-loop section).

echo "============================================================"
echo "§V.D Ablation Launch — REDUCED 12-cell sequential"
echo "Cell 1 pathB_seed42 is already running independently (PID 5666)"
echo "Started at: $(date)"
echo "Total cells to launch: ${#CONFIGS[@]}"
echo "============================================================"

mkdir -p logs/ablation

# ===========================================================================
# Wait for cell 1 to finish before starting cell 2
# ===========================================================================

CELL1_PID=5666
if ps -p $CELL1_PID > /dev/null 2>&1; then
    echo ""
    echo "Cell 1 (pathB_seed42, PID $CELL1_PID) still running. Waiting for completion..."
    echo "Will check every 60 sec."
    while ps -p $CELL1_PID > /dev/null 2>&1; do
        sleep 60
        CURRENT_STEP=$(tail -1 results/pathB_step6_seed42_20260513_165856/train_log.csv 2>/dev/null | cut -d, -f1)
        echo "  $(date '+%H:%M:%S') cell 1 step: $CURRENT_STEP"
    done
    echo "Cell 1 completed at $(date)"
fi

# ===========================================================================
# Helpers
# ===========================================================================

is_config_done() {
    local config="$1"
    # Search for any existing run directory with high-step ckpt
    # Match results/*<config>* (config name appears in output dir name)
    local dirs=$(ls -d results/*${config}* 2>/dev/null)
    for dir in $dirs; do
        if [ -d "$dir" ]; then
            local high_ckpt=$(ls "$dir"/ckpt_step00*.pt 2>/dev/null | awk -F'step' '{print $2}' | awk -F'.' '{print $1}' | sort -n | tail -1)
            if [ -n "$high_ckpt" ] && [ "$high_ckpt" -ge 30000 ] 2>/dev/null; then
                return 0
            fi
        fi
    done
    return 1
}

check_disk() {
    local free_gb=$(df --output=avail -BG /root/autodl-tmp | tail -1 | sed 's/G//' | tr -d ' ')
    if [ "$free_gb" -lt "$MIN_FREE_DISK_GB" ]; then
        echo "ABORT: free disk $free_gb GB < threshold $MIN_FREE_DISK_GB GB"
        return 1
    fi
    echo "  disk free: ${free_gb} GB"
    return 0
}

# ===========================================================================
# Main loop
# ===========================================================================

idx=0
skipped=0
launched=0
for config in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    echo ""
    echo "------------------------------------------------------------"
    echo "[$idx/${#CONFIGS[@]}] Config: $config"
    echo "Time: $(date)"

    if is_config_done "$config"; then
        echo "  → SKIP (already trained)"
        skipped=$((skipped + 1))
        continue
    fi

    if ! check_disk; then
        echo ""
        echo "============================================================"
        echo "ABORTED at $idx/${#CONFIGS[@]}"
        echo "Launched: $launched, Skipped: $skipped, Remaining: $((${#CONFIGS[@]} - idx + 1))"
        echo "============================================================"
        exit 2
    fi

    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"

    timestamp=$(date +%Y%m%d_%H%M%S)
    logfile="logs/ablation/${config}_${timestamp}.log"
    pidfile="logs/ablation/${config}.pid"

    echo "  → LAUNCH (log: $logfile)"
    nohup python -m otp.train.train_otp_soft \
        --config-name "$config" \
        > "$logfile" 2>&1 &
    pid=$!
    echo "$pid" > "$pidfile"
    echo "  PID: $pid (started $(date '+%H:%M:%S'))"

    wait $pid
    exit_code=$?
    end_time=$(date '+%H:%M:%S')

    if [ "$exit_code" -eq 0 ]; then
        echo "  → DONE at $end_time (exit 0)"
        launched=$((launched + 1))
    else
        echo "  → FAILED at $end_time (exit $exit_code)"
        echo ""
        echo "============================================================"
        echo "Training failure for $config"
        echo "Check log: $logfile"
        echo "Auto-aborting subsequent cells."
        echo "============================================================"
        exit 3
    fi
done

echo ""
echo "============================================================"
echo "ALL TRAINING CELLS COMPLETE"
echo "Launched: $launched, Skipped: $skipped, Total: ${#CONFIGS[@]}"
echo "Finished at: $(date)"
echo "============================================================"
echo ""
echo "Next steps (manual):"
echo "  1. Run LIBERO-Goal M3 cross-bench probe (~1 hr GPU):"
echo "     python scripts/diagnostic/05_libero_goal_m3.py  # (script to be created Day 6+)"
echo ""
echo "  2. Aggregate §V.D results:"
echo "     python scripts/diagnostic/25_v_d_ablation_aggregate.py \\"
echo "         --phase both --n-episodes-per-task 50 --cells-config reduced"
