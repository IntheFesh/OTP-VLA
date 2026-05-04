#!/usr/bin/env bash
# scripts/04b_openvla_frozen_1epoch.sh
#
# 1-epoch frozen-backbone training with the real OpenVLA-OFT checkpoint.
#
# Prerequisites:
#   - GPU with >= 24 GB VRAM (bf16 OpenVLA-7B at batch=16)
#   - Network access to HF mirror, or checkpoint already cached in $HF_HOME
#   - pip install transformers accelerate peft
#
# Env vars respected:
#   HF_ENDPOINT  — HuggingFace CDN (default: https://hf-mirror.com on AutoDL)
#   HF_HOME      — local cache dir  (default: /root/autodl-tmp/hf_cache)
#
# Output:
#   results/openvla_frozen_1ep/   — checkpoints + train_log.csv
#   logs/openvla_frozen_1ep.log   — combined stdout/stderr

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO_ROOT}/logs"

echo "============================================================"
echo " OpenVLA-OFT 1-epoch frozen training"
echo " HF_ENDPOINT : ${HF_ENDPOINT}"
echo " HF_HOME     : ${HF_HOME}"
echo " Output      : results/openvla_frozen_1ep"
echo "============================================================"

T0=$(date +%s)

python -m otp.train.train_otp_soft \
    --config-name otp_soft_frozen \
    model.backbone_mode=frozen \
    model.backbone_dim=4096 \
    train.num_epochs=1 \
    train.num_steps=null \
    train.batch_size=16 \
    train.precision=bf16 \
    train.lr=1e-4 \
    train.weight_decay=1e-4 \
    train.warmup_steps=100 \
    train.lr_schedule=cosine \
    train.grad_clip=1.0 \
    train.log_every=50 \
    train.ckpt_every=500 \
    output_dir=results/openvla_frozen_1ep \
    2>&1 | tee "${REPO_ROOT}/logs/openvla_frozen_1ep.log"

T1=$(date +%s)
echo ""
echo "Wall-clock: $((T1 - T0)) s"
echo "Log: ${REPO_ROOT}/logs/openvla_frozen_1ep.log"
