#!/bin/bash

# PDD: Pruning During Distillation - Complete Pipeline
# این اسکریپت تمام مراحل آموزش را به صورت خودکار اجرا می‌کند

echo "=========================================="
echo "PDD: Pruning During Distillation Pipeline"
echo "=========================================="
echo ""

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs

# Step 1: Pre-train Teacher Model (ResNet56)
echo "Step 1/3: Pre-training Teacher Model (ResNet56)..."
echo "This may take 2-3 hours..."
python pretrain_teacher.py \
    --epochs 150 \
    --batch-size 128 \
    --lr 0.1 \
    --save-dir checkpoints \
    | tee logs/teacher_training.log

if [ ! -f "checkpoints/resnet56_teacher_best.pth" ]; then
    echo "Error: Teacher model training failed!"
    exit 1
fi

echo ""
echo "Teacher model trained successfully!"
echo ""

# Step 2: Pruning During Distillation
echo "Step 2/3: Pruning During Distillation..."
echo "Training student model (ResNet20) with dynamic pruning..."
python train.py \
    --teacher-checkpoint checkpoints/resnet56_teacher_best.pth \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.01 \
    --temperature 4.0 \
    --alpha 0.5 \
    --save-dir checkpoints \
    | tee logs/pdd_training.log

if [ ! -f "checkpoints/best_masked_model.pth" ]; then
    echo "Error: PDD training failed!"
    exit 1
fi

echo ""
echo "PDD training completed successfully!"
echo ""

# Step 3: Fine-tune Pruned Model
echo "Step 3/3: Fine-tuning Pruned Model..."
echo "Fine-tuning the pruned model to recover accuracy..."
python finetune_pruned.py \
    --checkpoint checkpoints/pruning_plan.pth \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.01 \
    --save-dir checkpoints \
    | tee logs/finetuning.log

if [ ! -f "checkpoints/pruned_model_finetuned_best.pth" ]; then
    echo "Error: Fine-tuning failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "=========================================="
echo ""
echo "Results are saved in:"
echo "  - Teacher model: checkpoints/resnet56_teacher_best.pth"
echo "  - Masked model: checkpoints/best_masked_model.pth"
echo "  - Pruning plan: checkpoints/pruning_plan.pth"
echo "  - Final model: checkpoints/pruned_model_finetuned_best.pth"
echo ""
echo "Logs are saved in logs/ directory"
echo ""
