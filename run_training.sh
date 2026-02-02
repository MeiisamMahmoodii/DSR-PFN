#!/bin/bash

# ✅ SERVER TRAINING SCRIPT - 4x A100 GPU
# Status: PROVEN WORKING - All critical fixes applied
# Last validated: 2026-01-14

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --standalone main.py \
  --epochs 500 \
  --batch_size 1 \
  --lr 1e-4 \
  --min_vars 20 \
  --max_vars 50 \
  --num_layers 16 \
  --lambda_aux_moe 0.1 \
  --lambda_delta 1.0 \
  --grad_checkpoint \
  --grad_clip 10.0 \
  --loss_type focal \
  --intervention_prob 0.3

# To resume after interruption:
# Add --resume flag to above command
