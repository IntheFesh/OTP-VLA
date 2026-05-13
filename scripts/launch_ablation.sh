#!/bin/bash
# scripts/launch_ablation.sh
#
# Sequential launch of §V.D ablation retrains.
# Each config trained ~12.5 hr; total 21 (or 18) cells × 12.5 hr ≈ 11-12 days.
#
# Usage:
#   bash scripts/launch_ablation.sh full      # 21 cells (variantc included)
#   bash scripts/launch_ablation.sh no_variantc  # 18 cells (variantc dropped after decision gate fail)
#
# Features:
#   - Resume capability: skips cells whose results directory already has final ckpt
#   - Disk safety gate: aborts if free space < 50 GB before launching new cell
#   - PID tracking: writes each PID to logs/ablation/<config>.pid
#   - Progress logging: per-cell log file in logs/ablation/<config>_<timestamp>.log
#
# Pre-registration: notes/v_d_ablation_preregistration_2026-05-13.md

set -uo pipefail

REPO_ROOT="/root/autodl-tmp/OTP-VLA"
cd "$REPO_ROOT" || { echo "ERR: cannot cd to $REPO_ROOT"; exit 1; }

# ===========================================================================
# Configuration
# ===========================================================================

MIN_FREE_DISK_GB=50  # abort threshold

MODE="${1:-full}"
if [ "$MODE" != "full" ] && [ "$MODE" != "no_variantc" ]; then
    echo "ERR: usage: $0 [full|no_variantc]"
    exit 1
fi

# Build config list (matches notes/v_d_ablation_preregistration §1.1)
CONFIGS=(
    # Cell A: Path B base (3 seeds)
    "otp_soft_pathB_seed42"
    "otp_soft_pathB_seed137"
    "otp_soft_pathB_seed2026"
    # Cell B: Path B + Cocos (3 seeds)
    "otp_soft_ablation_pathB_cocos_seed42"
    "otp_soft_ablation_pathB_cocos_seed137"
    "otp_soft_ablation_pathB_cocos_seed2026"
    # Cell D: V3 baseline (3 seeds)
    "otp_soft_ablation_v3_seed42"
    "otp_soft_ablation_v3_seed137"
    "otp_soft_ablation_v3_seed2026"
    # Cell E: drop mesh (3 seeds)
    "otp_soft_ablation_drop_mesh_seed42"
    "otp_soft_ablation_drop_mesh_seed137"
    "otp_soft_ablation_drop_mesh_seed2026"
    # Cell F: drop grasp (3 seeds)
    "otp_soft_ablation_drop_grasp_seed42"
    "otp_soft_ablation_drop_grasp_seed137"
    "otp_soft_ablation_drop_grasp_seed2026"
    # Cell G: drop proprio (3 seeds)
    "otp_soft_ablation_drop_proprio_seed42"
    "otp_soft_ablation_drop_proprio_seed137"
    "otp_soft_ablation_drop_proprio_seed2026"
)

# Add Cell C: Variant (c) if full mode
if [ "$MODE" = "full" ]; then
    CONFIGS+=(
        "otp_soft_ablation_variantc_seed42"
        "otp_soft_ablation_variantc_seed137"
        "otp_soft_ablation_variantc_seed2026"
    )
fi

echo "============================================================"
echo "§V.D Ablation Launch — Mode: $MODE"
echo "Total configs: ${#CONFIGS[@]}"
echo "Started at: $(date)"
echo "============================================================"

mkdir -p logs/ablation

# ===========================================================================
# Helpers
# ===========================================================================

# Check if a config has already been trained (resume capability).
# A config is considered "done" if results/<config>_<ts>/ contains a final ckpt
# at step matching production end of training (~6600 steps × 50 epoch = ~33000 steps).
# We treat presence of ckpt_step0033000.pt (or higher) as completion marker.
is_config_done() {
    local config="$1"
    # Find latest run directory matching this config's basename pattern
    # Production configs save to results/{config_name_short}_{timestamp}/
    # Approximation: search for any directory containing config in name with ckpt > step 30000
    local dirs=$(ls -td results/*${config}* 2>/dev/null | head -5)
    for dir in $dirs; do
        if [ -d "$dir" ]; then
            # Look for high-step ckpt (≥30000 indicates near-end-of-training)
            local high_ckpt=$(ls "$dir"/ckpt_step00*.pt 2>/dev/null | awk -F'step' '{print $2}' | awk -F'.' '{print $1}' | sort -n | tail -1)
            if [ -n "$high_ckpt" ] && [ "$high_ckpt" -ge 30000 ] 2>/dev/null; then
                return 0  # done
            fi
        fi
    done
    return 1  # not done
}

# Check disk space
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

    # Resume check
    if is_config_done "$config"; then
        echo "  → SKIP (already trained; found ckpt ≥ step 30000)"
        skipped=$((skipped + 1))
        continue
    fi

    # Disk gate
    if ! check_disk; then
        echo ""
        echo "============================================================"
        echo "ABORTED at $idx/${#CONFIGS[@]}"
        echo "Configs done: $launched launched, $skipped resumed"
        echo "Configs remaining: $((${#CONFIGS[@]} - idx + 1))"
        echo "============================================================"
        exit 2
    fi

    # Set environment
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"

    # Launch with nohup, wait for completion (sequential)
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

    # Wait for this config to finish before launching next
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
        echo "Configs done: $launched launched, $skipped resumed"
        echo "Configs remaining: $((${#CONFIGS[@]} - idx))"
        echo "Continue or abort? Auto-aborting to be safe."
        echo "============================================================"
        exit 3
    fi
done

echo ""
echo "============================================================"
echo "ALL CONFIGS COMPLETE"
echo "Launched: $launched"
echo "Resumed/skipped: $skipped"
echo "Total: ${#CONFIGS[@]}"
echo "Finished at: $(date)"
echo "============================================================"
echo ""
echo "Next step: run §V.D analysis"
echo "  python scripts/diagnostic/25_v_d_ablation_aggregate.py \\"
echo "      --phase both --n-episodes-per-task 50 --cells-config $MODE"
