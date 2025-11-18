"""
Pruning script for PDD (Pruning During Distillation)
This file contains only the pruning phase

EXACT IMPLEMENTATION AS PER PAPER:
Paper states: "a score of 0 indicates that the channel is redundant and can be pruned, 
               otherwise, it is preserved."

Therefore: threshold = 0.0 (not 0.5)
- score = 0 → PRUNE
- score > 0 → KEEP
"""

import torch
import argparse
import os

from models.resnet import resnet20
from utils.pruner import ModelPruner
from utils.helpers import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Prune PDD Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint with masks')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save pruned model')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Pruning threshold (paper uses 0.0)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("Phase 2: Model Pruning (PDD Paper Implementation)")
    print("="*70)
    print(f"Threshold: {args.threshold} (paper: score=0 means prune)")
    print("="*70)
    
    # Check if checkpoint exists
    if not os.path.isfile(args.checkpoint):
        print(f"\n✗ Error: Checkpoint not found at {args.checkpoint}")
        print("Please run train.py first to generate the checkpoint with masks")
        exit(1)
    
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'masks' not in checkpoint:
        print("✗ Error: No masks found in checkpoint!")
        print("Please make sure the checkpoint is from training phase")
        exit(1)
    
    print("✓ Checkpoint loaded successfully")
    
    # Create student model
    print("\nCreating student model...")
    student = resnet20(num_classes=10)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        student.load_state_dict(checkpoint['state_dict'])
    else:
        student.load_state_dict(checkpoint)
    
    print("✓ Model weights loaded")
    
    # Get masks
    masks = checkpoint['masks']
    print(f"✓ Loaded {len(masks)} masks")
    
    # Display mask value distribution
    print("\n" + "-"*70)
    print("Mask Value Distribution:")
    print("-"*70)
    
    for name, mask in masks.items():
        values = mask.squeeze().cpu()
        num_zero = (values == 0.0).sum().item()
        num_nonzero = (values > 0.0).sum().item()
        total = values.numel()
        
        print(f"{name:40s} | Total: {total:4d} | "
              f"Zeros: {num_zero:4d} ({100*num_zero/total:5.2f}%) | "
              f"NonZeros: {num_nonzero:4d} ({100*num_nonzero/total:5.2f}%)")
        print(f"{'':40s} | Min: {values.min():.4f} | Max: {values.max():.4f} | "
              f"Mean: {values.mean():.4f} | Std: {values.std():.4f}")
    
    print("-"*70)
    
    # Display mask statistics with threshold
    print("\n" + "-"*70)
    print(f"Channel Pruning Statistics (threshold={args.threshold}):")
    print("-"*70)
    
    total_channels = 0
    kept_channels = 0
    
    for name, mask in masks.items():
        # Paper: "score of 0 indicates redundant" → use threshold=0.0
        mask_binary = (mask.squeeze() > args.threshold).float()
        total = mask_binary.numel()
        kept = mask_binary.sum().item()
        
        total_channels += total
        kept_channels += kept
        
        pruning_ratio = (1 - kept / total) * 100
        print(f"{name:40s} | Total: {total:4d} | Kept: {kept:4d} | "
              f"Pruned: {total-kept:4d} ({pruning_ratio:5.2f}%)")
    
    overall_pruning = (1 - kept_channels / total_channels) * 100
    print("-"*70)
    print(f"{'Overall':40s} | Total: {total_channels:4d} | "
          f"Kept: {kept_channels:4d} | "
          f"Pruned: {total_channels-kept_channels:4d} ({overall_pruning:5.2f}%)")
    print("-"*70)
    
    # Create pruner
    print("\nInitializing pruner...")
    pruner = ModelPruner(student, masks, threshold=args.threshold)
    
    # Calculate original statistics
    original_params, pruned_params_est = pruner.get_params_count()
    original_flops, pruned_flops_est = pruner.get_flops_count()
    
    print("\n" + "="*70)
    print("Compression Statistics (Estimated):")
    print("="*70)
    print(f"Original Parameters: {original_params:,}")
    print(f"Pruned Parameters:   {pruned_params_est:,}")
    print(f"Parameters Reduction: {(1 - pruned_params_est/original_params)*100:.2f}%")
    print(f"\nOriginal FLOPs: {original_flops:,}")
    print(f"Pruned FLOPs:   {pruned_flops_est:,}")
    print(f"FLOPs Reduction: {(1 - pruned_flops_est/original_flops)*100:.2f}%")
    print("="*70)
    
    # Prune the model
    print("\nPruning model...")
    print("-"*70)
    pruned_student = pruner.prune()
    print("-"*70)
    print("✓ Model pruned successfully")
    
    # Calculate actual statistics after pruning
    actual_params = sum(p.numel() for p in pruned_student.parameters())
    params_reduction = (1 - actual_params / original_params) * 100
    flops_reduction = (1 - pruned_flops_est / original_flops) * 100
    
    print("\n" + "="*70)
    print("Final Compression Statistics:")
    print("="*70)
    print(f"Original Parameters: {original_params:,}")
    print(f"Pruned Parameters:   {actual_params:,}")
    print(f"Parameters Reduction: {params_reduction:.2f}%")
    print(f"\nOriginal FLOPs: {original_flops:,}")
    print(f"Pruned FLOPs (estimated): {pruned_flops_est:,}")
    print(f"FLOPs Reduction: {flops_reduction:.2f}%")
    print("="*70)
    
    # Save pruned model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'pruned_student.pth')
    
    save_checkpoint({
        'state_dict': pruned_student.state_dict(),
        'params_reduction': params_reduction,
        'flops_reduction': flops_reduction,
        'original_params': original_params,
        'pruned_params': actual_params,
        'original_flops': original_flops,
        'pruned_flops': pruned_flops_est,
        'threshold': args.threshold,
        'source_checkpoint': args.checkpoint
    }, save_path)
    
    print(f"\n✓ Pruned model saved to: {save_path}")
    print("\nNext step: Run finetune.py to fine-tune the pruned model")
    print("="*70 + "\n")
    
    # Print paper comparison
    print("\n" + "="*70)
    print("PAPER RESULTS COMPARISON (ResNet20 student, ResNet56 teacher):")
    print("="*70)
    print("Paper results on CIFAR10:")
    print("  - FLOPs Reduction: 39.53%")
    print("  - Parameters Reduction: 32.77%")
    print("  - Accuracy: 91.65% (vs 91.48% baseline)")
    print(f"\nYour results:")
    print(f"  - FLOPs Reduction: {flops_reduction:.2f}%")
    print(f"  - Parameters Reduction: {params_reduction:.2f}%")
    print("  - Accuracy: [will be determined after fine-tuning]")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
