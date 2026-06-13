#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

# Example script for RL training of DiffusionAdapter

# Check arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <DIFFUSION_MODEL_PATH> <VAE_MODEL_PATH> <TARGET_MODEL_PATH> <AUTHENTICITY_MODEL_PATH>"
    echo ""
    echo "Example:"
    echo "  $0 ./models/diffusion_mlp5 ./models/vae ./models/target_lr.joblib ./models/auth_scorer/best_model.pt"
    exit 1
fi

DIFFUSION_MODEL_PATH=$1
VAE_MODEL_PATH=$2
TARGET_MODEL_PATH=$3
AUTHENTICITY_MODEL_PATH=$4

# Output directory
OUTPUT_DIR=./output/adapter_rl_$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "REINFORCE V3 Training for DiffusionAdapter"
echo "=========================================="
echo "Diffusion model: ${DIFFUSION_MODEL_PATH}"
echo "VAE model: ${VAE_MODEL_PATH}"
echo "Target model: ${TARGET_MODEL_PATH}"
echo "Authenticity model: ${AUTHENTICITY_MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Run training
python rl_training/algorithms/reinforce_v3.py \
    --diffusion_model_path ${DIFFUSION_MODEL_PATH} \
    --vae_model_path ${VAE_MODEL_PATH} \
    --target_model_path ${TARGET_MODEL_PATH} \
    --authenticity_model_path ${AUTHENTICITY_MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --latent_dim 1024 \
    --adapter_rank 128 \
    --max_delta 0.15 \
    --num_episodes 500 \
    --eval_every 20 \
    --save_every 50 \
    --batch_size 8 \
    --num_steps 30 \
    --max_new_tokens 128 \
    --lr 1e-5 \
    --entropy_coef 0.01 \
    --grad_clip_norm 0.5 \
    --replay_size 256 \
    --replay_ratio 0.25 \
    --candidate_multiplier 2 \
    --top_k 8 \
    --early_stop_patience 10 \
    --device cuda:0

echo ""
echo "Training complete!"
echo "Best adapter saved to: ${OUTPUT_DIR}/best_adapter.pt"
