#!/bin/bash

# Script to run PDD training

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p data

# Run training with default parameters
python main.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.01 \
    --temperature 4.0 \
    --alpha 0.5 \
    --finetune_epochs 100 \
    --finetune_lr 0.01 \
    --seed 42 \
    --device cuda

# Alternative: Run with smaller batch size for limited GPU memory
# python main.py --batch_size 128 --num_workers 2
