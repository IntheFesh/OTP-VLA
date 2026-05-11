#!/usr/bin/env bash
# =============================================================================
# Path B Step 6: Sequential multi-seed retrain launcher.
#
# Usage:
#   bash scripts/launch/launch_pathB_step6.sh [SEEDS...] [--cocos]
#
# Examples:
#   # Run all 3 seeds, base Path B (no Cocos)
#   bash scripts/launch/launch_pathB_step6.sh 42 137 2026
#
#   # Run all 3 seeds with Cocos enabled
#   bash scripts/launch/launch_pathB_step6.sh 42 137 2026 --cocos
#
#   # Run just seed 42
#   bash scripts/launch/launch_pathB_step6.sh 42
#
# Behavior:
#   - Sequential launch (next seed waits for prev to finish)
#   - Each seed logs to logs/pathB_step6_seed{N}_{timestamp}.log
#   - Survives SSH disconnect via nohup
#   - Pre-flight check: GPU memory > 60GB free (Stage 3 still using ~16GB)
#   - Aborts if previous seed crashes (non-zero exit)
# =============================================================================

set -e

REPO_ROOT="/root/autodl-tmp/OTP-VLA"
cd "$REPO_ROOT"

# Parse args
USE_COCOS=false
SEEDS=()
for arg in "$@"; do
    if [[ "$arg" == "--cocos" ]]; then
        USE_COCOS=true
    else
        SEEDS+=("$arg")
    fi
done

if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=(42 137 2026)
fi

echo "=============================================="
echo "Path B Step 6 multi-seed launcher"
echo "=============================================="
echo "Seeds: ${SEEDS[*]}"
echo "Cocos: $USE_COCOS"
echo "Repo:  $REPO_ROOT"
echo

# Pre-flight: GPU memory check
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Pre-flight GPU memory free: ${FREE_MB} MiB"
if [[ "$FREE_MB" -lt 60000 ]]; then
    echo "WARNING: < 60GB GPU memory free. May OOM with Stage 3 still running."
    echo "Continue anyway? (Ctrl+C to abort, Enter to continue)"
    read -r
fi

# Cocos CLI override
COCOS_OVERRIDE=""
if [[ "$USE_COCOS" == "true" ]]; then
    COCOS_OVERRIDE="model.decoder.use_cocos_source=true"
fi

for SEED in "${SEEDS[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="logs/pathB_step6_seed${SEED}_${TIMESTAMP}.log"
    CONFIG_NAME="otp_soft_pathB_seed${SEED}"

    echo "[$(date '+%H:%M:%S')] Launching seed $SEED..."
    echo "  Config: $CONFIG_NAME"
    echo "  Log:    $LOG_FILE"
    echo "  Cocos:  $USE_COCOS"

    # Launch in foreground (this script runs in nohup outer layer)
    if [[ -n "$COCOS_OVERRIDE" ]]; then
        python -u -m otp.train.train_otp_soft \
            --config-name="$CONFIG_NAME" \
            $COCOS_OVERRIDE \
            > "$LOG_FILE" 2>&1
    else
        python -u -m otp.train.train_otp_soft \
            --config-name="$CONFIG_NAME" \
            > "$LOG_FILE" 2>&1
    fi

    EXIT_CODE=$?
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[$(date '+%H:%M:%S')] FAIL: seed $SEED exited with $EXIT_CODE"
        echo "Last 30 lines of log:"
        tail -30 "$LOG_FILE"
        echo
        echo "Aborting multi-seed launcher. Remaining seeds not started."
        exit 1
    fi

    echo "[$(date '+%H:%M:%S')] DONE: seed $SEED finished successfully"
    echo

    # Brief pause between seeds for GPU memory cleanup
    sleep 30
done

echo "=============================================="
echo "All seeds completed."
echo "=============================================="
echo
echo "Next: run aggregation"
echo "  python scripts/diagnostic/20_pathB_multiseed_aggregate.py"
